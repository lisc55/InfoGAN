mkdir('data');
for  i=1:100000
    tic;
    gen_random_head(i);
    toc;
end