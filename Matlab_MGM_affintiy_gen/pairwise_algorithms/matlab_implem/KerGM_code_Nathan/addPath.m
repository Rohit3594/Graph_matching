function addPath
% Add folders of predefined functions into matlab searching paths.
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 03-20-2009
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-07-2013
%   modify  -  Nathan Buskulic (nathan.buskulic@outlook.fr), 03-25-2020

global footpath;
footpath = cd;

addpath(genpath([footpath '/src']));
addpath(genpath([footpath '/lib']));
addpath('./KerGMExact');