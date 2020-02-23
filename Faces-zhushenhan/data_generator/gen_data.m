mkdir('data');
for  i=1:10000
    tic;
    gen_random_head(i);
    toc;
end