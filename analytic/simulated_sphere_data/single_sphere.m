clc; 
% Mie theory
clear;clc;
nb=1.4;
num_mults=10;

for radius =100:200:200
    op =bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
    enei = linspace( 400, 700, 201 );
    nmsqrd_to_micronsqrd = (10^(-7))^2;
    mie = miesolver( epstable( ['gold.dat'] ), epsconst( nb^2 ),  2*(radius), op, 'lmax', num_mults);    
    ext = mie.ext( enei )*nmsqrd_to_micronsqrd;
    sca = mie.sca( enei )*nmsqrd_to_micronsqrd;
    plot( enei, sca , '-'  );  hold on;

%     filename = strcat('Sph',num2str(radius),'nm_BEMMIE_l',num2str(num_mults),'_n',num2str(nb),'.mat');
%     save(filename, 'enei', 'ext', 'sca');
end


%% asdf

clc; 
% Mie theory
clear;clc;clf;
nb=1.473;
radius = 50;

num_mults=1;
op =bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
enei = linspace( 400, 700, 201 );
nmsqrd_to_micronsqrd = (10^(-7))^2;
mie = miesolver( epstable( ['gold.dat'] ), epsconst( nb^2 ),  2*(radius), op, 'lmax', num_mults);    
ext = mie.ext( enei )*nmsqrd_to_micronsqrd;
sca = mie.sca( enei )*nmsqrd_to_micronsqrd;
% plot( enei, sca/max(sca) , '-'  ); hold on;
abs = ext-sca;
% plot( enei, abs , '-'  ); hold on;

num_mults=10;
op =bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
enei = linspace( 400, 700, 201 );
nmsqrd_to_micronsqrd = (10^(-7))^2;
mie = miesolver( epstable( ['gold.dat'] ), epsconst( nb^2 ),  2*(radius), op, 'lmax', num_mults);    
ext = mie.ext( enei )*nmsqrd_to_micronsqrd;
sca = mie.sca( enei )*nmsqrd_to_micronsqrd;
plot( enei, sca, '-'  ); hold on;
abs = ext-sca;
% plot( enei, abs , '-'  ); hold on;


