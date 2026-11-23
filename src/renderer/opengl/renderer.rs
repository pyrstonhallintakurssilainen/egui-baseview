use baseview::{PhySize, Window};
use egui::FullOutput;
use egui_glow::Painter;
use std::sync::Arc;

use super::OpenGlError;

/// Windows: the current OpenGL context is per thread, and a plug-in editor renders on the
/// host's main thread. If the editor left its own context current (or, after
/// `make_not_current`, none at all), a host that renders with OpenGL on the same thread
/// (e.g. an eframe application) would draw its next frame into the wrong context and its UI
/// would fall apart. So the editor saves the context that was current before it renders and
/// restores it afterwards. The same applies when the editor closes: its objects are deleted
/// in its own context, not the host's, whose textures and buffers with the same names would
/// otherwise be deleted.
#[cfg(target_os = "windows")]
mod wgl {
    use std::ffi::c_void;

    #[link(name = "opengl32")]
    extern "system" {
        fn wglGetCurrentContext() -> *mut c_void;
        fn wglGetCurrentDC() -> *mut c_void;
        fn wglMakeCurrent(dc: *mut c_void, rc: *mut c_void) -> i32;
    }

    /// Device and context handles (as addresses, so the type is `Copy` without raw pointers).
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Current {
        dc: usize,
        rc: usize,
    }

    /// The thread's current context (`rc` = 0 if none is current).
    pub fn current() -> Current {
        // SAFETY: plain queries about the calling thread.
        unsafe {
            Current {
                dc: wglGetCurrentDC() as usize,
                rc: wglGetCurrentContext() as usize,
            }
        }
    }

    impl Current {
        pub fn is_some(&self) -> bool {
            self.rc != 0
        }

        /// Makes this context current (an empty one releases the current context). Returns
        /// whether it succeeded.
        ///
        /// # Safety
        /// The handles must be alive (or empty), and the call must be made on the same thread.
        pub unsafe fn make(&self) -> bool {
            wglMakeCurrent(self.dc as *mut c_void, self.rc as *mut c_void) != 0
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphicsConfig {
    /// Controls whether to apply dithering to minimize banding artifacts.
    ///
    /// Dithering assumes an sRGB output and thus will apply noise to any input value that lies between
    /// two 8bit values after applying the sRGB OETF function, i.e. if it's not a whole 8bit value in "gamma space".
    /// This means that only inputs from texture interpolation and vertex colors should be affected in practice.
    ///
    /// Defaults to true.
    pub dithering: bool,

    /// Needed for cross compiling for VirtualBox VMSVGA driver with OpenGL ES 2.0 and OpenGL 2.1 which doesn't support SRGB texture.
    /// See <https://github.com/emilk/egui/pull/1993>.
    ///
    /// For OpenGL ES 2.0: set this to [`egui_glow::ShaderVersion::Es100`] to solve blank texture problem (by using the "fallback shader").
    pub shader_version: Option<egui_glow::ShaderVersion>,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            shader_version: None,
            dithering: true,
        }
    }
}

pub struct Renderer {
    glow_context: Arc<egui_glow::glow::Context>,
    painter: Painter,
    /// The editor's own context (Windows), so its objects are deleted there (`Drop`).
    #[cfg(target_os = "windows")]
    own: wgl::Current,
}

impl Renderer {
    pub fn new(window: &Window, config: GraphicsConfig) -> Result<Self, OpenGlError> {
        let context = window.gl_context().ok_or(OpenGlError::NoContext)?;
        #[cfg(target_os = "windows")]
        let prev = wgl::current();
        unsafe {
            context.make_current();
        }
        #[cfg(target_os = "windows")]
        let own = wgl::current();

        #[allow(clippy::arc_with_non_send_sync)]
        let glow_context = Arc::new(unsafe {
            egui_glow::glow::Context::from_loader_function(|s| context.get_proc_address(s))
        });

        let painter = egui_glow::Painter::new(
            Arc::clone(&glow_context),
            "",
            config.shader_version,
            config.dithering,
        )
        .map_err(OpenGlError::CreatePainter);

        // Give the host its context back even if creating the painter failed.
        #[cfg(target_os = "windows")]
        restore(prev, own, context);
        #[cfg(not(target_os = "windows"))]
        unsafe {
            context.make_not_current();
        }
        let painter = painter?;

        Ok(Self {
            glow_context,
            painter,
            #[cfg(target_os = "windows")]
            own,
        })
    }

    pub fn max_texture_side(&self) -> usize {
        self.painter.max_texture_side()
    }

    pub fn render(
        &mut self,
        window: &Window,
        bg_color: egui::Rgba,
        physical_size: PhySize,
        pixels_per_point: f32,
        egui_ctx: &mut egui::Context,
        full_output: &mut FullOutput,
    ) {
        let PhySize {
            width: canvas_width,
            height: canvas_height,
        } = physical_size;

        let shapes = std::mem::take(&mut full_output.shapes);
        let textures_delta = &mut full_output.textures_delta;

        let context = window
            .gl_context()
            .expect("failed to get baseview gl context");
        #[cfg(target_os = "windows")]
        let prev = wgl::current();
        unsafe {
            context.make_current();
        }

        unsafe {
            use egui_glow::glow::HasContext as _;
            self.glow_context
                .clear_color(bg_color.r(), bg_color.g(), bg_color.b(), bg_color.a());
            self.glow_context.clear(egui_glow::glow::COLOR_BUFFER_BIT);
        }

        for (id, image_delta) in &textures_delta.set {
            self.painter.set_texture(*id, image_delta);
        }

        let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);
        let dimensions: [u32; 2] = [canvas_width, canvas_height];

        self.painter
            .paint_primitives(dimensions, pixels_per_point, &clipped_primitives);

        for id in textures_delta.free.drain(..) {
            self.painter.free_texture(id);
        }

        unsafe {
            context.swap_buffers();
        }
        #[cfg(target_os = "windows")]
        restore(prev, self.own, context);
        #[cfg(not(target_os = "windows"))]
        unsafe {
            context.make_not_current();
        }
    }
}

/// Restores the context that was current before rendering (the host's). If there was none,
/// or it was the editor's own, releases the editor's context as before.
#[cfg(target_os = "windows")]
fn restore(prev: wgl::Current, own: wgl::Current, context: &baseview::gl::GlContext) {
    // SAFETY: `prev` was current on this thread right before rendering, so it is still alive.
    unsafe {
        if prev.is_some() && prev != own && prev.make() {
            return;
        }
        context.make_not_current();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Windows: delete the objects in the editor's own context and give the host its
        // context back. If our context cannot be made current (window already destroyed),
        // skip the deletion: destroying the context frees the objects, and deleting them in the
        // wrong context would hit the host's objects.
        #[cfg(target_os = "windows")]
        {
            let prev = wgl::current();
            // SAFETY: `own` is the editor's context; if it is already gone, `make` fails.
            unsafe {
                if self.own.is_some() && self.own.make() {
                    self.painter.destroy();
                }
                if prev != self.own {
                    prev.make();
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        self.painter.destroy()
    }
}
