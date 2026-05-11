import matplotlib.pyplot as plt

# 1. 准备数据 (根据你的表格录入)
thresholds = [0.6, 0.63, 0.65, 0.67, 0.7]
w_f1 = [0.8457, 0.8400, 0.8700, 0.8480, 0.8448]
w_pre = [0.8847, 0.8609, 0.9023, 0.8897, 0.8713]
w_rec = [0.8100, 0.8200, 0.8400, 0.8100, 0.8200]
nil_rec = [66.67, 71.79, 71.79, 66.67, 71.79]

# 2. 设置画布风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

# --- 绘制左轴: 综合指标 ---
line1, = ax1.plot(thresholds, w_f1, marker='o', color='#1f77b4', lw=3, label='Weighted F1')
line2, = ax1.plot(thresholds, w_pre, marker='s', color='#2ca02c', lw=1.5, ls='--', label='Weighted Precision')
line3, = ax1.plot(thresholds, w_rec, marker='d', color='#ff7f0e', lw=1.5, ls='--', label='Weighted Recall')

ax1.set_xlabel('Threshold (阈值)', fontsize=12, fontweight='bold')
ax1.set_ylabel('综合指标得分', fontsize=12, fontweight='bold')
ax1.set_ylim(0.80, 0.92) # 根据数据范围设定
ax1.grid(True, ls=':', alpha=0.6)

# --- 绘制右轴: NIL 专项指标 ---
ax2 = ax1.twinx()
line4, = ax2.plot(thresholds, nil_rec, marker='^', color='#d62728', lw=2, ls='-.', label='NIL 召回率 (%)')
ax2.set_ylabel('NIL 召回率 (%)', fontsize=12, color='#d62728', fontweight='bold')
ax2.set_ylim(60, 80) # 专项指标的刻度范围

# 3. 整合图例
lns = [line1, line2, line3, line4]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower right', frameon=True, fontsize=10)

# 4. 标注峰值 (0.65, 0.8700)
ax1.annotate(f'最佳 F1: 0.8700\n(Threshold=0.65)', xy=(0.65, 0.8700), xytext=(0.66, 0.88),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

plt.title('Impact of LLM Threshold on Name Disambiguation Performance', fontsize=14, pad=20)
plt.tight_layout()

# 5. 保存并显示
# plt.savefig('threshold_analysis.pdf') # 建议保存为 PDF 格式用于论文打印
plt.show()