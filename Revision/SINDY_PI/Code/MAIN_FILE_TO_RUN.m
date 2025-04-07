%% Adapted code from 
%% Original source:
%% https://github.com/dynamicslab/SINDy-PI/tree/master/DoublePendulum

% Loading data from python
load('dp.mat')
load('d_dp.mat')  %derivative, caluclated in python (DP_generator.py   DoublePendulumDerivArray)
load('dp2.mat')   % Different initial values:  DoublePendulumTrajectory(initial_theta1=numpy.pi/5,initial_theta2=0.1,initial_omega1=-1.2,initial_omega2=0.5,
                  %                                                         timestep=0.05,numstep=10000,l1=1+1*.15,l2=1,m1=0.25*3,m2=2.5,g=9.81)
load('d_dp2.mat')  %derivative of the differnet initial values 

my_Data = [struct.theta1; struct.theta2; struct.omega1; struct.omega2]';
my_d_Data = [d_struct.d_theta1; d_struct.d_theta2; d_struct.d_omega1; d_struct.d_omega2]';

my_Data_test = [struct2.theta1; struct2.theta2; struct2.omega1; struct2.omega2]';
my_dData_test =[d_struct2.d_theta1; d_struct2.d_theta2; d_struct2.d_omega1; d_struct2.d_omega2]';

tspan =struct.t;
tspan_test = struct2.t;

state0_test = [struct2.theta1(1); struct2.theta2(1); struct2.omega1(1); struct2.omega2(1)]';
Control=0;u=0;Shuffle = 0;
%% Add some noise to the system
Noise = 0;
for i=1:4
    my_Data(:,i)=my_Data(:,i)+Noise*randn(size(my_Data(:,i)));
end
%
for i=1:4
    my_d_Data(:,i)=my_d_Data(:,i)+Noise*randn(size(my_d_Data(:,i)));
end

%% Now perform sparse regression of non-linear dynamics

% Get the number of states we have
[dtat_length,n_state]=size(my_Data);

% Define the control input(Should be zero in our example)
n_control=0;

% Choose whether you want to display actual ODE or not
disp_actual_ode=1;

% If the ODEs you want to display is the actual underlyting dynamics of the
% system, please set actual as 1
actual=1;

% Create symbolic states
dz=sym('dz',[n_state,1]);

% Now we first create the parameters of the function right hand side
Highest_Poly_Order_Guess=1;
Highest_Trig_Order_Guess=1;
Highest_U_Order_Guess=0;

% Then create the right hand side library parameters
Highest_Poly_Order=1;
Highest_Trig_Order=4;
Highest_U_Order=0;
Highest_dPoly_Order=1;
%% Define parameters for the sparese regression
lam=[1e-4;5e-4;1e-3;2e-3;3e-3;4e-3;5e-3;6e-3;7e-3;8e-3;9e-3;1e-2;2e-2;3e-2;4e-2;5e-2;...
    6e-2;7e-2;8e-2;9e-2;1e-1;2e-1;3e-1;4e-1;5e-1;6e-1;7e-1;8e-1;9e-1;1;1.5;2;2.5;3;3.5;4;4.5;5;...
    6;7;8;9;10;20;30;40;50;100;200];

N_iter=20;
disp=0;
NormalizeLib=0;

for iter=1:n_state
    fprintf('\n \n Calculating the %i expression...\n',iter)
    
    % According to the previous parameter generate the left hand side guess
    [LHS_Data,LHS_Sym]=GuessLib(my_Data,my_d_Data(:,iter),iter,u,Highest_Poly_Order_Guess,Highest_Trig_Order_Guess,Highest_U_Order_Guess);
    
    %Generate the corresponding data
    [SINDy_Data,SINDy_Struct]=SINDyLib(my_Data,my_d_Data(:,iter),iter,u,Highest_Poly_Order,Highest_Trig_Order,Highest_U_Order,Highest_dPoly_Order);
    
    % Run the for loop and try all the left hand guess
    for i=1:length(LHS_Sym)
        if iter==1 && i==1
            Xi=cell(n_state,length(LHS_Sym),length(lam));
            ODE=cell(n_state,length(LHS_Sym),length(lam));
            ODEs=cell(n_state,length(LHS_Sym),length(lam));
        end

        % Print the left hand side that we are testing
        fprintf('\t Testing the left hand side as %s:\n',char(LHS_Sym{i}))
        
        % Exclude the guess from SINDy library
        [RHS_Data,RHS_Struct]=ExcludeGuess(SINDy_Data,SINDy_Struct,LHS_Sym{i});
        
        parfor j=1:length(lam)
            % Select the sparse threashold
            lambda=lam(j);
    
            % Perform the sparse regression problem
            [Xi{iter,i,j},ODE{iter,i,j}]=sparsifyDynamics(RHS_Data,LHS_Data(:,i),LHS_Sym{i},lambda,N_iter,RHS_Struct,disp,NormalizeLib);
            
            % Perform sybolic calculation and solve for dX
            digits(6)
            ODE_Guess=vpa(solve(LHS_Sym{i}==ODE{iter,i,j},dz(iter)));
            
            % Print the discovered ODE
            fprintf(strcat('\t The corresponding ODE we found is: ',char(dz(iter,1)),'=',char((ODE_Guess)),'\n \n'));
            
            % Store the result
            ODEs{iter,i,j}=ODE_Guess;
        end
    end
end
%% Now generate the ODE function file and test the accuracy of the
% identified system

fprintf('\v Start calculating the best model that could represent the training data...\n \n')
for iter=1:n_state
    % Print which expression are you working on
    fprintf('\t Calculating the best model for the %d expression...\n',iter)
    
    for i=1:length(LHS_Sym)
        % Print the process
        fprintf('\t Calculating the score of previously found ODE on the test data, %d %% finished. \n',round((i/length(LHS_Sym))*100))
        
        for j=1:length(lam)
            % If the previous ODE is 0, set the score as NaN, else calculate
            % it.
            if isempty(ODEs{iter,i,j})
                ODE_Not_Exist=1;
                Score(iter,i,j)=NaN;
            else
                % Generate the ODE file
                Generate_ODE_RHS(ODEs{iter,i,j},n_state,n_control);
                % Calculate the accuracy of the file
                Score(iter,i,j)=Get_Score(my_dData_test(:,iter),my_Data_test,u,Control,tspan_test,state0_test,Shuffle);
            end
        end
        
        % Get the best lambda
        [minVal1(iter,i),minIndex1(iter,i)]=min(Score(iter,i,:));
        
    end
    
    % Get the best score and use this ODE file
    [minVal2(iter,1),minIndex2(iter,1)]=min(minVal1(iter,:));
    
    % Store the best ODE
    ODE_Best(iter,1)=ODEs{iter,minIndex2(iter,1),minIndex1(iter,minIndex2(iter,1))};
    
    % Print the Result
    fprintf('\n\n\n\t The SINDy-PI discovered Best ODE for the %d expression is:\n',iter)
    fprintf('\t %s = %s \n\n\n',char(dz(iter)),char((ODE_Best(iter,1)))')
end

%% Now generate this best guess ODE and print its result
disp_best=1;
if disp_best==1
    fprintf('\n\n\n\v The SINDy-PI discovered Best ODE for the whole system is:\n')
    digits(4)
    for iter=1:n_state
        fprintf(strcat('\v ******\v\t',char(dz(iter)),'=',char(simplify(ODE_Best(iter,1))),'\n'));
    end
end
%% Get the simulation result
% Generate the ODE file
fprintf('\n\n\n\v Generating the Best Model for comparision...\n')
Generate_ODE_RHS(ODE_Best(:,1),n_state,n_control);

% Define the vairables for be able to simulate and compare
Noise_test=0;
% original state
state0 = [struct.theta1(1); struct.theta2(1); struct.omega1(1); struct.omega2(1)];
tspan = struct.t;
[dData_Es,Data_Es]=Get_Sim_Data(@(t,z)Sindy_ODE_RHS(t,z,u),state0,u,tspan,Noise_test,Control,Shuffle);



%% Plotting 
close all
figure(1)
plot(tspan_test,my_Data(:,1),'linewidth',1,'Color','black' )
hold on
plot(tspan_test,Data_Es(:,1),'linewidth',1,'linestyle','--','color','blue')
legend("Real", "Approximate")

save('predicted', 'Data_Es')


