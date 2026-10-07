% livelink_rails.m - solve every exported rail in COMSOL (LiveLink for MATLAB) and record
% the nucleation time and the wall-clock per rail.  No GUI model is needed: every rail is built from
% scratch through the API (1-D geometry with one interval per segment, Coefficient Form PDE).
%
%   export folder:  set STACKEM_CR (default <home>/stackem_work/outputs/comsol); on Windows for example
%                   setenv('STACKEM_CR', 'C:\stackem_comsol\comsol')   before running this script
%   inputs:         rails.csv, constants.json         (comsol/export_rails.py)
%   output:         comsol_rails_result.csv           rail, n_seg, t_nuc_comsol_s, seconds_build, seconds_solve, seconds_total
%   optional:       set STACKEM_CR_MAX to solve only the first N rails (a timing sample), default all
%
% Start "COMSOL Multiphysics with MATLAB", cd to the folder of this file, then:  livelink_rails
% This is the version that produced the released result (COMSOL 6.2).  Two points differ from what one would
% write from the paper's notation; both were settled by the actual run:
%   * the dependent variable keeps COMSOL's default name u (the field-renaming line below has no effect in 6.2),
%     so the initial value and the maximum operator are written with u;
%   * the coefficient-form flux is  -c*grad(u) + ga ; with c = kap the Korhonen flux -kap*(u_x - G - M) needs
%     ga = +kap*(G+M)  (G, M as defined in stackem/korhonen_fdm.py, Z_eff = +10).
% Property names follow COMSOL 6.x; if your version complains about a property, check the name with
% model.<node>.properties in the MATLAB console (the meaning is documented next to every line).
import com.comsol.model.*; import com.comsol.model.util.*;
cr = getenv('STACKEM_CR'); if isempty(cr), cr = fullfile(getenv('HOME'), 'stackem_work/outputs/comsol'); end
nmax = str2double(getenv('STACKEM_CR_MAX')); if isnan(nmax), nmax = inf; end
tab = readtable(fullfile(cr, 'rails.csv'));
C = jsondecode(fileread(fullfile(cr, 'constants.json')));
kB = 1.380649e-23; e0 = 1.602176634e-19;
tout = 10 .^ (4:0.1:9.5);                                     % same output grid as the reference solver export
rails = unique(tab.rail)'; rails = rails(1:min(numel(rails), nmax));
fid = fopen(fullfile(cr, 'comsol_rails_result.csv'), 'w');
fprintf(fid, 'rail,n_seg,t_nuc_comsol_s,seconds_build,seconds_solve,seconds_total\n');
for ri = rails
    rows = tab(tab.rail == ri, :); S = height(rows);
    t_all = tic;
    % ---------------- geometry: one interval per segment (shared points = junctions) ----------------
    model = ModelUtil.create('Model'); comp = model.component.create('comp1', true);
    geom = comp.geom.create('geom1', 1);
    x0 = [0; cumsum(rows.L_m)];                                % junction coordinates (m)
    iv = geom.create('i1', 'Interval'); iv.set('intervals', 'many'); iv.set('p', sprintf('%.12g ', x0));   % 'p' = interval end points
    geom.run;
    % ---------------- per-segment variables: T(x), dT/dx, kappa, G, M, sigma_T ----------------
    for k = 1:S
        L = rows.L_m(k); Gam = rows.Gamma_m(k); lam = L / Gam;
        TL = rows.T_L_K(k); TR = rows.T_R_K(k); Tm = rows.T_m_K(k); j = rows.j_A_per_m2(k); sT = rows.sigma_T_Pa(k);
        v = comp.variable.create(sprintf('var%d', k)); v.selection.geom('geom1', 1); v.selection.set(k);
        v.set('xi',   sprintf('(x - %.12g[m]) / %.12g[m]', x0(k), L));
        v.set('T',    sprintf('%.9g[K] + (%.9g[K]) * xi + %.9g[K] * (1 - cosh((xi - 0.5) * %.9g) / cosh(%.9g))', TL, TR - TL, Tm, lam, 0.5 * lam));
        v.set('dTdx', sprintf('(%.9g[K]) / %.12g[m] - %.9g[K] / %.12g[m] * sinh((xi - 0.5) * %.9g) / cosh(%.9g)', TR - TL, L, Tm, Gam, lam, 0.5 * lam));
        v.set('kap',  sprintf('%.9g[m^2/s] * exp(-%.9g[eV] / (%.9g[J/K] * T)) * %.9g[Pa] * %.9g[m^3] / (%.9g[J/K] * T)', C.D0, C.Ea_eV, kB, C.B, C.Omega, kB));
        v.set('G',    sprintf('%.9g[C] * %.9g * %.9g[ohm*m] * (%.9g[A/m^2]) / %.9g[m^3]', e0, C.Z_eff, C.rho, j, C.Omega));
        v.set('M',    sprintf('%.9g[eV] / (%.9g[m^3] * T) * dTdx', C.Q_eV, C.Omega));
        v.set('sT',   sprintf('%.9g[Pa]', sT));
    end
    % ---------------- Korhonen equation as a Coefficient Form PDE; dependent variable u = sigma [Pa] ----------------
    pde = comp.physics.create('c', 'CoefficientFormPDE', 'geom1'); pde.field('dimensionless').field('sigma');
    pde.prop('Units').set('DependentVariableQuantity', 'none'); pde.prop('Units').set('CustomDependentVariableUnit', 'Pa');
    pde.prop('Units').set('SourceTermQuantity', 'none');       pde.prop('Units').set('CustomSourceTermUnit', 'Pa/s');
    eq = pde.feature('cfeq1');
    eq.set('c', 'kap'); eq.set('da', 1); eq.set('a', 0); eq.set('f', 0); eq.set('ea', 0); eq.set('al', 0); eq.set('be', 0);
    eq.set('ga', 'kap*(G+M)');                                % conservative flux source: flux = -kap*u_x + ga  ->  zero flux = blocked ends
    pde.feature('init1').set('u', 'sT');                      % initial stress = residual stress of the segment (u is the stress in Pa)
    % ---------------- mesh: 60 elements per segment (the reference solver's resolution) ----------------
    mesh = comp.mesh.create('mesh1'); ed = mesh.create('edg1', 'Edge'); sz = ed.create('size1', 'Size');
    sz.set('custom', 'on'); sz.set('hmaxactive', true); sz.set('hmax', sprintf('%.12g', min(rows.L_m) / 60)); mesh.run;
    % ---------------- maximum over the rail, time-dependent study ----------------
    mx = comp.cpl.create('maxop1', 'Maximum'); mx.selection.geom('geom1', 1); mx.selection.all;
    std = model.study.create('std1'); tm = std.create('time', 'Transient');
    tm.set('tlist', '10^range(4,0.1,9.5)'); tm.set('rtolactive', true); tm.set('rtol', '1e-6');
    t_build = toc(t_all); t_solve = tic;
    std.run;
    t_solve = toc(t_solve);
    % ---------------- nucleation time: first crossing of sigma_crit, linear in sqrt(t) (as the reference) ----------------
    smax = mphglobal(model, 'maxop1(u)', 'dataset', 'dset1', 't', tout);
    k = find(smax >= C.sigma_crit, 1);
    if isempty(k), tn = inf;
    elseif k == 1, tn = tout(1);
    else, s0 = sqrt(tout(k-1)); s1 = sqrt(tout(k)); f = (C.sigma_crit - smax(k-1)) / (smax(k) - smax(k-1)); tn = (s0 + f * (s1 - s0))^2; end
    t_total = toc(t_all);
    fprintf(fid, '%d,%d,%.9e,%.3f,%.3f,%.3f\n', ri, S, tn, t_build, t_solve, t_total);
    fprintf('rail %4d: %3d segments  t_nuc %.3f yr  build %.2fs solve %.2fs\n', ri, S, tn / 3.15576e7, t_build, t_solve);
    ModelUtil.remove('Model');
end
fclose(fid); fprintf('wrote %s\n', fullfile(cr, 'comsol_rails_result.csv'));
