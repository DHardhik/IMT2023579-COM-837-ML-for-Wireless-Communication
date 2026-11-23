%% gen_channels.m
% =========================================================================
% COM837 Assignment 3 - STEP 1: Ground-truth channel generation
%
% Generates SISO time-frequency channel grids H (72 subcarriers x 14 OFDM
% symbols) for the VehA and SUI5 models using the FastFading class of the
% Vienna 5G Link-Level Simulator (+Channel/FastFading.m).
%
% Output (one file per model, in ./data):
%   H_VehA.mat / H_SUI5.mat with
%     Hr, Hi : single, [72 x 14 x N]  real / imaginary parts of H
%     cfg    : struct with every simulation parameter (for the report)
%
% Pilots, noise, LS/MMSE and the deep networks are done later in Python;
% this script only produces the PERFECT channel H (the training target).
%
% Run from the project root (the folder that contains 5GLLSimulator/).
% =========================================================================
clear; clc;
% Locate the project root from this file's own location (robust to run()/cd)
projRoot = fileparts(fileparts(mfilename('fullpath')));   % .../MLWC_A3
simDir   = fullfile(projRoot, '5GLLSimulator');
% handle a doubly-nested unzip (5GLLSimulator/5GLLSimulator/...)
if exist(fullfile(simDir, '5GLLSimulator', '+Channel'), 'dir')
    simDir = fullfile(simDir, '5GLLSimulator');
end
if ~exist(fullfile(simDir, '+Channel', 'FastFading.m'), 'file')
    error('FastFading.m not found. Expected: %s', fullfile(simDir, '+Channel'));
end
addpath(simDir);       % parent of +Channel must be on the path for Channel.FastFading
cd(projRoot);
if ~exist('data', 'dir'), mkdir('data'); end

%% ------------------------- configuration --------------------------------
N_FRAMES   = 40000;     % full dataset used in the report (set 1000 for a quick test)
                        %     (paper: 32000 train / 4000 val / 4000 test)

cfg.NS     = 72;        % subcarriers  (LTE 1.4 MHz: 6 RBs x 12)
cfg.ND     = 14;        % OFDM symbols per subframe (normal CP)
cfg.df     = 15e3;      % subcarrier spacing [Hz]
cfg.Nfft   = 256;       % FFT size -> sampling rate 256*15 kHz = 3.84 MHz
cfg.fs     = cfg.df * cfg.Nfft;
cfg.Ncp    = 18;        % ~4.69 us normal CP at 3.84 MHz
cfg.fc     = 2.1e9;     % carrier frequency [Hz]   (paper)
cfg.v_kmh  = 50;        % UE speed [km/h]          (paper)
cfg.fD     = cfg.v_kmh/3.6 * cfg.fc / 2.998e8;     % max Doppler ~ 97 Hz
cfg.nPaths = 100;       % sinusoids per tap for the Jakes sum-of-sinusoids
cfg.subsamp= 4;         % time sub-sampling inside FastFading (speed-up);
                        % 4 samples = 1 us << coherence time (~4 ms)

Nsym          = cfg.Nfft + cfg.Ncp;                 % samples per OFDM symbol
SamplesTotal  = cfg.ND * Nsym;                      % samples per subframe
% channel is sampled in the middle of the useful part of every OFDM symbol
cfg.TimePos   = (0:cfg.ND-1)*Nsym + cfg.Ncp + cfg.Nfft/2;
cfg.ActiveSC  = 1:cfg.NS;                           % 72 contiguous subcarriers

%% ----------------------- channel models ---------------------------------
% VehA is built into FastFading. SUI5 is NOT, so it is passed as a
% self-defined power-delay-profile vector (linear power per sample).
% SUI-5 (omni): delays [0 4 10] us, powers [0 -5 -10] dB.
sui5_delay = [0 4e-6 10e-6];
sui5_pdB   = [0 -5 -10];
idx        = round(sui5_delay * cfg.fs) + 1;        % -> samples 1, 16, 39
sui5_pdp   = zeros(1, max(idx));
sui5_pdp(idx) = 10.^(sui5_pdB/10);

models = { 'VehA', 'VehicularA', 1 ;               % name, PDP, rng seed
           'SUI5', sui5_pdp,     2 };

%% ----------------------- generation loop --------------------------------
for m = 1:size(models,1)
    name = models{m,1};  pdp = models{m,2};  seed = models{m,3};
    rng(seed);                                      % reproducibility

    ch = Channel.FastFading( ...
        cfg.fs, pdp, SamplesTotal, cfg.fD, 'Jakes', cfg.nPaths, ...
        false, ...            % correlatedFrames: independent subframes
        'none', 0, 0, ...     % no spatial correlation (SISO)
        1, 1, ...             % 1 Tx, 1 Rx antenna
        false, ...            % no sample-rate warning
        0, 0, ...             % TWDP K, delta (unused for fading)
        struct('spatialChannelModel', false), ...
        [1 1], [1 1], cfg.fc, cfg.subsamp);

    fprintf('%s: RMS delay spread = %.2f us\n', name, ch.GetRmsDelaySpread*1e6);

    Hr = zeros(cfg.NS, cfg.ND, N_FRAMES, 'single');
    Hi = zeros(cfg.NS, cfg.ND, N_FRAMES, 'single');
    tic;
    for n = 1:N_FRAMES
        ch.NewRealization(n);                       % new independent channel
        % H(k,l) = FFT of the impulse response at time of symbol l
        Hf = ch.GetTransferFunction(cfg.TimePos, cfg.Nfft, cfg.ActiveSC);
        Hr(:,:,n) = single(real(Hf));
        Hi(:,:,n) = single(imag(Hf));
        if mod(n, 2000) == 0
            fprintf('  %s %d/%d  (%.0f s)\n', name, n, N_FRAMES, toc);
        end
    end
    pw = mean(Hr(:).^2 + Hi(:).^2);
    fprintf('%s done in %.0f s, mean |H|^2 = %.3f (should be ~1)\n', name, toc, pw);

    cfgOut = cfg; cfgOut.model = name; cfgOut.seed = seed; cfgOut.N = N_FRAMES;
    if ~ischar(pdp), cfgOut.pdp = pdp; end
    save(fullfile('data', ['H_' name '.mat']), 'Hr', 'Hi', 'cfgOut', '-v7.3');

    % quick look: one channel image (real / imag) -> sanity check, like Fig. 1
    figure('Name', name);
    subplot(1,2,1); imagesc(Hr(:,:,1)); axis xy; colorbar;
    title([name ' - Real(H)']); xlabel('OFDM symbol'); ylabel('Subcarrier');
    subplot(1,2,2); imagesc(Hi(:,:,1)); axis xy; colorbar;
    title([name ' - Imag(H)']); xlabel('OFDM symbol'); ylabel('Subcarrier');
end
