function gen_golden_m2_wavespeed(which, avds)
%GEN_GOLDEN_M2_WAVESPEED  Reference hex wave speeds at the frozen M2 settings.
%
%   Runs the verbatim reference implementations (repro/ref/{flat,junction})
%   for the homing drive at two frozen conventions:
%     - "paper":  flat 300x300, checkpoints 50%/60% of n  (paper p.6)
%     - "code":   flat 200x200, checkpoints 40%/70% of n  (published launcher)
%     - junction runs use actual_length L, checkpoints L/(2*sqrt(3))+0.45L /
%       +0.65L (published junction launcher)
%
%   Also records the *effective* kernel mean distance sum(w*d)/sum(w) for each
%   dispersal value, which exposes the low-avd lattice discretisation noted by
%   the M0 evaluator.
%
%   Usage: gen_golden_m2_wavespeed()                 % all runs
%          gen_golden_m2_wavespeed({'flat_paper'})   % subset

    here = fileparts(mfilename('fullpath'));
    hexroot = fileparts(here);
    refroot = fullfile(here, 'ref');
    goldendir = fullfile(hexroot, 'golden');
    if ~isfolder(goldendir); mkdir(goldendir); end
    if nargin < 1 || isempty(which)
        which = {'flat_paper', 'flat_code', 'junction_code', 'junction_paper'};
    end

    work = tempname();
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's'));
    copyfile(fullfile(refroot, 'flat'), fullfile(work, 'flat'));
    copyfile(fullfile(refroot, 'junction'), fullfile(work, 'junction'));

    gcr = 0; dc = 1; herr = 0; ddfitness = 1;
    if nargin < 2 || isempty(avds)
        avds = [0.5, 1.0, 2.0];
    end

    rows = struct('label', {}, 'kind', {}, 'avd', {}, 'speed', {}, ...
                  'm', {}, 'n', {}, 'checkpoint1', {}, 'checkpoint2', {}, ...
                  'kernel_mean_dist', {}, 'elapsed_s', {});

    fd = fullfile(work, 'flat');
    jd = fullfile(work, 'junction');

    for k = 1:numel(which)
        label = which{k};
        switch label
            case 'flat_paper', kind = 'flat'; m = 300; n = 300; cp = [round(0.5*n), round(0.6*n)];
            case 'flat_code',  kind = 'flat'; m = 200; n = 200; cp = [round(0.4*n), round(0.7*n)];
            case 'junction_code',  kind = 'junction'; L = 600;
            case 'junction_paper', kind = 'junction'; L = 300;
            otherwise, error('unknown label %s', label);
        end

        if strcmp(kind, 'flat')
            cd(fd); addpath(fd); addpath(fullfile(fd, 'homing'));
            [drive_name, dp] = drive_generator(gcr, dc, herr, ddfitness);
            save(fullfile(fd, 'homing', 'dp.mat'), 'dp');
            mode_params = [0, cp(1), cp(2)];
        else
            cd(jd); addpath(jd); addpath(fullfile(jd, 'homing'));
            [drive_name, dp] = drive_generator(gcr, dc, herr, ddfitness);
            save(fullfile(jd, 'homing', 'dp.mat'), 'dp');
            cp = [round(L / (2 * sqrt(3)) + 0.45 * L), round(L / (2 * sqrt(3)) + 0.65 * L)];
            m = ceil(L * (1 + 1 / sqrt(3))); n = ceil(L * 2 / sqrt(3));
            mode_params = [0, cp(1), cp(2)];
        end

        for ai = 1:numel(avds)
            avd = avds(ai); sigma = avd / sqrt(pi / 2);
            [K, mean_dist] = kernel_stats(sigma);
            t0 = tic;
            if strcmp(kind, 'flat')
                [~, speed] = evalc('main(m, n, 0, 0, mode_params, drive_name, sigma)');
            else
                [~, speed] = evalc('main(L, 0, 0, mode_params, drive_name, sigma)');
            end
            el = toc(t0);
            rows(end + 1) = struct('label', label, 'kind', kind, 'avd', avd, ...
                'speed', speed, 'm', m, 'n', n, 'checkpoint1', cp(1), ...
                'checkpoint2', cp(2), 'kernel_mean_dist', mean_dist, 'elapsed_s', el);
            fprintf('%-15s avd=%-4.2f kind=%-8s m=%d n=%d cp=[%d %d] speed=%.6f mean_d=%.4g (%.1fs)\n', ...
                label, avd, kind, m, n, cp(1), cp(2), speed, mean_dist, el);
        end
        if strcmp(kind, 'flat')
            rmpath(fd); rmpath(fullfile(fd, 'homing')); clear main get_mig_matrix25;
        else
            rmpath(jd); rmpath(fullfile(jd, 'homing')); clear main get_mig_matrix25;
        end
    end
    cd(here);

    meta = struct();
    meta.created_by = 'hexagon_spatial_test/repro/gen_golden_m2_wavespeed.m';
    meta.model = 'general hex model (flat/junction), homing drive, boundary1=0.2, lambda=5, 51x51 kernel';
    meta.avds = avds;
    meta.unit_note = 'speed in cells/generation; flat is scaled by sqrt(3)/2 by wavespeed_analyse.m';
    meta.reference_files = dir_hashes(refroot);
    meta.runs = rows;

    save(fullfile(goldendir, 'hex_homing_wavespeed_m2.mat'), 'rows', 'meta');
    fid = fopen(fullfile(goldendir, 'hex_homing_wavespeed_m2.json'), 'w');
    fwrite(fid, jsonencode(meta, 'PrettyPrint', true));
    fclose(fid);
    fprintf('wrote golden/hex_homing_wavespeed_m2.{mat,json}\n');
end

function [K, mean_dist] = kernel_stats(sigma)
    K = get_mig_matrix25(sigma);
    len = size(K, 1); c = (len + 1) / 2;
    [ii, jj] = ndgrid(1:len, 1:len);
    dx = ii - c; dy = jj - c;
    d = sqrt(dx.^2 + dy.^2 - dx .* dy);
    mean_dist = sum(K .* d, 'all') / sum(K, 'all');
end

function s = dir_hashes(refroot)
    s = struct();
    dirs = {'flat', 'junction'};
    files = {'main.m', 'get_mig_matrix25.m', ...
             fullfile('homing', 'drive_generator.m'), ...
             fullfile('homing', 'renew_function.m')};
    for di = 1:numel(dirs)
        for fi = 1:numel(files)
            p = fullfile(refroot, dirs{di}, files{fi});
            key = matlab.lang.makeValidName([dirs{di} '_' strrep(files{fi}, filesep, '_')]);
            s.(key) = sha256(p);
        end
    end
end

function h = sha256(path)
    fid = fopen(path, 'r');
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    h = lower(reshape(dec2hex(uint8(typecast(md.digest(), 'uint8')), 2)', 1, []));
end
