choiceRisk = zeros(5,12);

choiceAmbig = zeros(5,12);


screensize = get(groot, 'Screensize');
fig = figure('Position', [screensize(3)/4 screensize(4)/22 screensize(3)/2 screensize(4)*10/12]);
ax1 = subplot(2,1,1);

% actual choice
barplot = bar(ax1,choiceRisk(1,:),'FaceColor','y');
hold on
xticklabels({'r25-$5','r25-$8','r25-$12','r25-$25','r50-$5','r50-$8','r50-$12','r50-$25','r75-$5','r75-$8','r75-$12','r75-$25'})
ylim([0 1.1])
yticks([0:0.1:1.1])

% predicted by model
plot(ax1,choiceRisk(2,:),'LineStyle','none','Marker','o','Color','blue','MarkerSize',8,'LineWidth',1 )
plot(ax1,choiceRisk(3,:),'LineStyle','none','Marker','+','Color','red','MarkerSize',8,'LineWidth',1)
plot(ax1,choiceRisk(4,:),'LineStyle','none','Marker','*','Color','green','MarkerSize',8,'LineWidth',1 )
plot(ax1,choiceRisk(5,:),'LineStyle','none','Marker','d','Color','black','MarkerSize',8,'LineWidth',1 )
legend({'Actual Choice','Model1 Arb','Model1 Rating','Model2 Arb','Model2 Rating'},'FontSize',12)

title(ax1,'Subject 2585 medical risky choice probability')

ax2 = subplot(2,1,2);
% actual choice
barplot = bar(ax2,choiceAmbig(1,:),'FaceColor','y');
hold on
xticklabels({'a24-$5','a24-$8','a24-$12','a24-$25','a50-$5','a50-$8','a50-$12','a50-$25','a74-$5','a74-$8','a74-$12','a74-$25'})
ylim([0 1.1])
yticks([0:0.1:1.1])

% predicted by model
plot(ax2,choiceAmbig(2,:),'LineStyle','none','Marker','o','Color','blue','MarkerSize',8,'LineWidth',1 )
plot(ax2,choiceAmbig(3,:),'LineStyle','none','Marker','+','Color','red','MarkerSize',8,'LineWidth',1 )
plot(ax2,choiceAmbig(4,:),'LineStyle','none','Marker','*','Color','green','MarkerSize',8,'LineWidth',1 )
plot(ax2,choiceAmbig(5,:),'LineStyle','none','Marker','d','Color','black','MarkerSize',8,'LineWidth',1 )

title(ax2,'Subject 2585 medical ambiguous choice probability')
    
