function rp = defrp
error(nargchk(0, 0, nargin));
rp.width        = 128;
rp.height       = 128;
rp.gamma        = 0;
rp.theta        = 0;
rp.phi          = 0;
rp.alpha        = 0;
rp.t2d          = [0;0];
rp.camera_pos   = [0;0;-3400];
rp.scale_scene  = 0.0;
rp.object_size  = 0.615 * 512;
rp.shift_object = [0;0;-46125];
rp.shift_world  = [0;0;0];
rp.scale        = 0.001;
rp.ac_g         = [1; 1; 1];
rp.ac_c         = 1;
rp.ac_o         = [0; 0; 0];
rp.ambient_col  = 0.6*[1; 1; 1];
rp.rotm         = eye(3);
rp.use_rotm     = 0;
rp.do_remap = 0; 
rp.dir_light = [];
rp.do_specular = 0.1;
rp.phong_exp = 8;
rp.specular = 0.1*255;
rp.do_cast_shadows = 1;
rp.sbufsize = 200;
rp.proj = 'perspective';
rp.f = 6000;
rp.n_chan = 3;
rp.backface_culling = 2;
rp.illum_method = 'phong';
rp.global_illum.brdf = 'lambert';
rp.global_illum.envmap = struct([]);
rp.global_illum.light_probe = [];
rp.ablend = [];
