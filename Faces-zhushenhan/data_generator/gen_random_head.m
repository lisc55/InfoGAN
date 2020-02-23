function gen_random_head(number)
[model msz] = load_model();
load attributes.mat
alpha = randn(msz.n_shape_dim, 1);
beta  = randn(msz.n_tex_dim, 1);
fat = unifrnd(-100,100);
shape  = coef2object( alpha+fat*weight_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV);
tex    = coef2object( beta+fat*weight_tex(1:msz.n_tex_dim),  model.texMU,   model.texPC,   model.texEV );
rp     = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = 0.6*ones(3,1);
phi = rand()*180-90;
light_phi = rand()*180-90;
gamma = rand()*40-20;
display_face(shape, tex, model.tl, rp, number, phi, gamma, light_phi);
