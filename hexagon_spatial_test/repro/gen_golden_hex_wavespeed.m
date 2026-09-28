function gen_golden_hex_wavespeed()
%GEN_GOLDEN_HEX_WAVESPEED  Generate small-scale hex-model wave-speed goldens.
%
%   Runs the verbatim reference implementations vendored under
%   repro/ref/{flat,junction} (from the published Hex-model repository,
%   parameter_sensitive/hex code上交版.zip -> "wave justification") on a
%   small homing-drive case, and writes results to golden/.
%
%   The reference model is the *general hex model* (Section "General hex
%   model" of the paper): a discrete-generation, continuous-count recursion
%       d_ar = rct*lambda./((lambda-1)*N+1)./N - ar.*N
%   followed by an imfilter with a 51x51 Gaussian kernel on the hexagonal
%   metric d = sqrt(dx^2 + dy^2 - dx*dy), 'replicate' boundary.
%
%   Usage:  matlab -batch "run('.../repro/gen_golden_hex_wavespeed.m')"
%   or      gen_golden_hex_wavespeed

    here = fileparts(mfilename('fullpath'));
    hexroot = fileparts(here);                 % hexagon_spatial_test/
    refroot = fullfile(here, 'ref');
    goldendir = fullfile(hexroot, 'golden');
    if ~isfolder(goldendir); mkdir(goldendir); end

    work = tempname();
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's'));
    copyfile(fullfile(refroot, 'flat'), fullfile(work, 'flat'));
    copyfile(fullfile(refroot, 'junction'), fullfile(work, 'junction'));

    gcr = 0; dc = 1; herr = 0; ddfitness = 1;
    avds = [0.25, 0.5, 1.0];

    flat = struct('direction', {}, 'avd', {}, 'speed', {}, 'm', {}, 'n', {}, ...
                  'checkpoint1', {}, 'checkpoint2', {}, 'elapsed_s', {});
    junc = struct('direction', {}, 'avd', {}, 'speed', {}, 'actual_length', {}, ...
                  'm', {}, 'n', {}, 'checkpoint1', {}, 'checkpoint2', {}, 'elapsed_s', {});

    % ---- flat side: wave travels along the x (second parameter) axis ----
    fd = fullfile(work, 'flat');
    cd(fd);
    addpath(fd); addpath(fullfile(fd, 'homing'));
    [drive_name, dp] = drive_generator(gcr, dc, herr, ddfitness);
    save(fullfile(fd, 'homing', 'dp.mat'), 'dp');
    m = 60; n = 200;
    cp1 = round(0.4 * n); cp2 = round(0.7 * n);
    mode_params = [0, cp1, cp2];
    for k = 1:numel(avds)
        avd = avds(k); sigma = avd / sqrt(pi / 2);
        t0 = tic;
        [~, speed] = evalc('main(m, n, 0, 0, mode_params, drive_name, sigma)');
        el = toc(t0);
        flat(end + 1) = struct('direction', 'flat', 'avd', avd, 'speed', speed, ...
            'm', m, 'n', n, 'checkpoint1', cp1, 'checkpoint2', cp2, 'elapsed_s', el);
        fprintf('flat     avd=%-5.3f m=%d n=%d cp=[%d %d] speed=%.6f (%.2fs)\n', ...
            avd, m, n, cp1, cp2, speed, el);
    end
    rmpath(fd); rmpath(fullfile(fd, 'homing')); clear main get_mig_matrix25;

    % ---- junction side: rotated domain, wave travels along the y axis ----
    jd = fullfile(work, 'junction');
    cd(jd);
    addpath(jd); addpath(fullfile(jd, 'homing'));
    [drive_name, dp] = drive_generator(gcr, dc, herr, ddfitness);
    save(fullfile(jd, 'homing', 'dp.mat'), 'dp');
    L = 100;
    cp1j = round(L / (2 * sqrt(3)) + 0.45 * L);
    cp2j = round(L / (2 * sqrt(3)) + 0.65 * L);
    mode_params = [0, cp1j, cp2j];
    for k = 1:numel(avds)
        avd = avds(k); sigma = avd / sqrt(pi / 2);
        t0 = tic;
        [~, speed] = evalc('main(L, 0, 0, mode_params, drive_name, sigma)');
        el = toc(t0);
        mj = ceil(L * (1 + 1 / sqrt(3))); nj = ceil(L * 2 / sqrt(3));
        junc(end + 1) = struct('direction', 'junction', 'avd', avd, 'speed', speed, ...
            'actual_length', L, 'm', mj, 'n', nj, ...
            'checkpoint1', cp1j, 'checkpoint2', cp2j, 'elapsed_s', el);
        fprintf('junction avd=%-5.3f L=%d m=%d n=%d cp=[%d %d] speed=%.6f (%.2fs)\n', ...
            avd, L, mj, nj, cp1j, cp2j, speed, el);
    end
    rmpath(jd); rmpath(fullfile(jd, 'homing'));

    meta = struct();
    meta.created_by = 'hexagon_spatial_test/repro/gen_golden_hex_wavespeed.m';
    meta.model = 'general hex model (flat/junction), homing drive, boundary1=0.2, lambda=5, kernel=51x51';
    meta.unit_note = 'speed in cells/generation; wavespeed_analyse.m scales flat-side speeds by sqrt(3)/2';
    meta.reference_files = dir_hashes(refroot);
    meta.flat = flat;
    meta.junction = junc;

    cd(here);
    save(fullfile(goldendir, 'hex_homing_wavespeed_smallcase.mat'), 'flat', 'junc', 'meta');
    fid = fopen(fullfile(goldendir, 'hex_homing_wavespeed_smallcase.json'), 'w');
    fwrite(fid, jsonencode(meta, 'PrettyPrint', true));
    fclose(fid);
    fprintf('wrote golden/hex_homing_wavespeed_smallcase.{mat,json}\n');
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
    digest = typecast(md.digest(), 'uint8');
    h = lower(reshape(dec2hex(uint8(digest), 2)', 1, []));
end
