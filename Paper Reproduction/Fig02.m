clearvars; close all; clc;

a = linspace(0,1,1000)';

for i=1:length(a)
    p = [a(i); 1-a(i)];
    q = [1-a(i); a(i)];
    E(i,1) = bsi(p,q);
end

plot(a,E,'-b',"LineWidth",2)
xlabel("$\alpha$","Interpreter","latex")
ylabel("$BSI(\alpha)$","Interpreter","latex")
box on
grid on
set(gca,"FontSize",12)
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 3, 3], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])