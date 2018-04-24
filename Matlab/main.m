clc
clear all
close all
A = csvread('A.csv')
B = csvread('B.csv')
Error = csvread('Error.csv')

Vertices = []
for j = 1:2 
    for i=1:6
        point=zeros(1,6)
        point(i)=Error(j,i)
        Vertices = [Vertices;point]
    end
end
W = Polyhedron(1.15 * Vertices);
W.computeHRep()

[Ke,~,~] = dlqr(A,B, eye(6), eye(2));

eig(A)

eig(A-B*Ke)
%%
rho = 0.05;
r_selected = Compute_mRPI(A-B*Ke, W, rho)
%%
BasisVector = W.V'
NumBasic = size(BasisVector,2)
%%
SampledPoint = []

for i = 1:5000
    Point = zeros(6, 1);
    for j = 1:r_selected
        Random = rand(NumBasic,1);
        w = BasisVector*Random/sum(Random);
        Point =  (A-B*Ke)*Point + w;
    end
    SampledPoint = [SampledPoint, Point];
end
SampledPoint

%%
Invariant = Polyhedron(1/(1-rho) * SampledPoint');
%%
Invariant.computeHRep()
Invariant.contains(zeros(6,1))
%%
Checking = []
for i = 1:1000
    Random = rand(size(SampledPoint,2),1);
    Normalized = Random/sum(Random);
    TestPoint = SampledPoint*Normalized;
    
    for j = 1:r_selected
        if Invariant.contains(TestPoint)==1
            Random = rand(NumBasic,1);
            w = BasisVector*Random/sum(Random);
            TestPoint =  (A-B*Ke)*TestPoint + w;
            Checking = [Checking; Invariant.contains(TestPoint)];
        else
            break
        end
    end
end
(sum(Checking))/size(Checking,1)

%%
upp_d = sdpvar(6,1)
low_d = sdpvar(6,1)

Constraints = []
for i = 1:size(SampledPoint,2)
    Constraints = [Constraints;
                  upp_d >= SampledPoint(:,i);
                  low_d <= SampledPoint(:,i)];
end
Cost = 0
for i = 1:6
    Cost = Cost + abs(upp_d(i,1)-low_d(i,1));
end
Problem = solvesdp(Constraints,Cost);

Upper_d = double(upp_d);
Lower_d = double(low_d);

%%
Checking1 = []
for i = 1:1000
    %Random = rand(size(SampledPoint,2),1);
    %Normalized = Random/sum(Random);
    %TestPoint = SampledPoint*Normalized;
    
    Random = rand(2,1);
    Normalized = Random/sum(Random);
    TestPoint = [Upper_d, Lower_d]*Normalized;
    
    for j = 1:r_selected
        if sum((Lower_d<=TestPoint)&(TestPoint<=Upper_d)) == size(TestPoint,1)
            Random = rand(NumBasic,1);
            w = BasisVector*Random/sum(Random);
            TestPoint =  (A-B*Ke)*TestPoint + w;
            Checking1 = [Checking1; sum((Lower_d<=TestPoint)&(TestPoint<=Upper_d)) == size(TestPoint,1)];
        else
            break
        end
    end
end
(sum(Checking1))/size(Checking1,1)


%%
save('Workspace')
[(sum(Checking))/size(Checking,1) (sum(Checking1))/size(Checking1,1)]