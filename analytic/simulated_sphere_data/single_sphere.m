clc;% clf;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

%  table of dielectric functions
epstab = { epsconst( 1.473^2 ), epstable( 'gold.dat' ) };

for radius = 5:5:100
    %  diameter of sphere
    diameter = 2*radius

    %  initialize sphere
    p = comparticle( epstab, { trisphere( 144, diameter ) }, [ 2, 1 ], 1, op );

    %  set up BEM solver
    bem = bemsolver( p, op );

    %  plane wave excitation
    exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0], op );

    %  light wavelength in vacuum
    enei = linspace( 400, 700, 200 );

    %  allocate scattering and extinction cross sections
    sca = zeros( length( enei ), 1 );
    ext = zeros( length( enei ), 1 );

    %  loop over wavelengths
    for ien = 1 : length( enei ) 
      %  surface charge
      sig = bem \ exc( p, enei( ien ) );
      %  scattering and extinction cross sections
      sca( ien, : ) = exc.sca( sig );
      ext( ien, : ) = exc.ext( sig );
    end

    nmsqrd_to_micronsqrd = (10^(-6));
    sca = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));
    ext= reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));

%     en_ev = 1240./enei; 
%     plot(enei, abs_mcsqrd); hold on;
    filename = strcat('Sph',num2str(radius),'nm_JC_ret_BEM_n1.4.mat');
    save(filename, 'enei', 'ext', 'sca');
end


%%  comparison with Mie theory
clear;clc;
for radius = 5:100:10
    op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
    enei = linspace( 400, 700, 200 );
    nmsqrd_to_micronsqrd = (10^(-6));
    
    mie = miesolver( epstable( 'gold.dat' ), epsconst( 1.473^2 ),  2*(radius), op,'lmax',1);    
    ext = mie.ext( enei )*nmsqrd_to_micronsqrd;
    sca = mie.sca( enei )*nmsqrd_to_micronsqrd;
    plot( enei, sca/max(sca), '-'  );  hold on;
    plot( enei, ext/max(ext), '-'  );  hold on;

    legend( 'sca', 'ext' );

%     filename = strcat('Sph',num2str(radius),'nm_JC_ret_l1_n1.473.mat');
%     save(filename, 'enei', 'ext', 'sca');
end
