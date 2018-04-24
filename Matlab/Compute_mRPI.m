function [ r_selected] = Compute_mRPI(Phi_e, W, rho)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%
for r = 1:1000
    Mat = Phi_e^r;
    Set = Mat*W;
    DisturbanceSet = rho * W;
    if (Set<= DisturbanceSet)
        r_selected = r
        break
    end;
end

end