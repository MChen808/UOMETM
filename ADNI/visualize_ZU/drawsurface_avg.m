load('visualize_ZU.mat')

lthick = year_60_left(idx, :);
rthick = year_60_right(idx, :);

%% Left
[nvertex, ntris, nconns, triloc, tris] = loadbyu('lh_white_regular_7.byu');

surf_l.vertices=triloc;
surf_l.faces=tris;
signal_l=lthick;

figure;
subplot(2,2,3);
figure_trimesh_noheadlight(surf_l, signal_l,'jet'); %left 
view([90 20]); caxis([0 4]); camlight(50,0);camlight(-50,0)
% title(idx);
colorbar off;

subplot(2,2,1);
figure_trimesh_noheadlight(surf_l, signal_l,'jet'); %left 
view([-90 20]); caxis([0 4]); camlight(50,0);camlight(-50,0)
% title('Left');
colorbar off;

%% Right
[nvertex, ntris, nconns, triloc, tris] = loadbyu('rh_white_regular_7.byu');

surf_r.vertices=triloc;
surf_r.faces=tris;
signal_r=rthick;

subplot(2,2,4);
figure_trimesh_noheadlight(surf_r,signal_r,'jet'); %right
view([-90 20]); caxis([0 4]); camlight(50,0); camlight(-50,0)
% title('Right'); 
colorbar off;

subplot(2,2,2);;
figure_trimesh_noheadlight(surf_r,signal_r,'jet'); %right
view([90 20]); caxis([0 4]); camlight(50,0);camlight(-50,0)
colorbar off;
% title('Right'); 
% 
% 
% view:  change view angle
% 角度可以直接在Matlab gui 的 figure 旋轉
% 若角度調不出來，可能要x y 軸互換 (or  xz or  yz),  改 surf_l.vertices
% 
% caxis: change color range