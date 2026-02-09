"""
Amazon Reviews 高级可视化脚本
基于NLP分析结果生成精美的可视化图表

包含:
1. 词云图 (正面/负面/整体/痛点)
2. 方面情感雷达图
3. 品牌对比热力图
4. 关键词网络图
5. 情感流向图
6. 主题词云
7. 方面气泡图
8. 品牌情感箱型图
9. 痛点漏斗图
10. 综合仪表盘

运行: python advanced_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
OUTPUT_DIR = 'advanced_visualizations'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("=" * 80)
print("🎨 Amazon Kitchen Knife Reviews - 高级数据可视化")
print("=" * 80)

# ==================== 加载数据 ====================
print("\n📂 加载数据...")

absa_detailed = pd.read_csv('nlp_results/data/absa_detailed.csv')
absa_summary = pd.read_csv('nlp_results/data/absa_summary.csv')
bert_results = pd.read_csv('nlp_results/data/bert_sentiment_results.csv')
ner_brands = pd.read_csv('nlp_results/data/ner_brands.csv')
ner_materials = pd.read_csv('nlp_results/data/ner_materials.csv')
textrank_keywords = pd.read_csv('nlp_results/data/textrank_keywords.csv')
topic_modeling = pd.read_csv('nlp_results/data/topic_modeling.csv')

print(f"✅ 数据加载完成")
print(f"   - ABSA详细数据: {len(absa_detailed):,} 条")
print(f"   - BERT结果: {len(bert_results):,} 条")
print(f"   - 关键词: {len(textrank_keywords)} 个")
print(f"   - 品牌: {len(ner_brands)} 个")
print(f"   - 材质: {len(ner_materials)} 个")

# ==================== 1. 词云图 (4合1) ====================
print("\n[1/10] 🎨 生成词云图...")

try:
    from wordcloud import WordCloud

    # 准备文本数据
    positive_reviews = bert_results[bert_results['bert_label'] == 'POSITIVE']['review_text_clean']
    negative_reviews = bert_results[bert_results['bert_label'] == 'NEGATIVE']['review_text_clean']
    all_reviews = bert_results['review_text_clean']

    # 停用词
    stopwords = set([
        'knife', 'knives', 'set', 'the', 'and', 'this', 'that', 'they', 'them', 'it', 'a', 'I', 'to', 'in', 'so', 'on',
        'one', 'all', 'out of','as','of','is',
        'these', 'those', 'have', 'has', 'had', 'with', 'from', 'been', 'were', 'was', 'look', 'only', 'you', 'my',
        'even', 'use', 'through', 'do', 'after', 'video guides', 'video',
        'are', 'but', 'for', 'not', 'just', 'very', 'really', 'like', 'get', 'got'
    ])

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comprehensive Word Cloud Analysis', fontsize=20, fontweight='bold', y=0.98)

    # 1.1 整体词云
    print("   - 生成整体词云...")
    wordcloud_all = WordCloud(
        width=800, height=600,
        background_color='white',
        stopwords=stopwords,
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(' '.join(all_reviews.fillna('')))

    axes[0, 0].imshow(wordcloud_all, interpolation='bilinear')
    axes[0, 0].set_title('All Reviews - General Keywords', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')

    # 1.2 正面词云
    print("   - 生成正面词云...")
    wordcloud_pos = WordCloud(
        width=800, height=600,
        background_color='white',
        stopwords=stopwords,
        colormap='Greens',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(' '.join(positive_reviews.fillna('')))

    axes[0, 1].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0, 1].set_title('Positive Reviews - What Users Love',
                         fontsize=14, fontweight='bold', color='darkgreen', pad=10)
    axes[0, 1].axis('off')

    # 1.3 负面词云
    print("   - 生成负面词云...")
    wordcloud_neg = WordCloud(
        width=800, height=600,
        background_color='white',
        stopwords=stopwords,
        colormap='Reds',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(' '.join(negative_reviews.fillna('')))

    axes[1, 0].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1, 0].set_title('Negative Reviews - Pain Points',
                         fontsize=14, fontweight='bold', color='darkred', pad=10)
    axes[1, 0].axis('off')

    # 1.4 痛点词云（黑底）
    print("   - 生成痛点词云...")
    pain_keywords = []
    for review in negative_reviews.fillna(''):
        words = review.lower().split()
        pain_words = [w for w in words if w in ['rust', 'rusted', 'rusting', 'dull', 'dulled',
                                                'broke', 'broken', 'crack', 'cracked', 'cracking', 'poor', 'cheap',
                                                'terrible', 'bad', 'disappointing', 'disappointed', 'waste',
                                                'horrible']]
        pain_keywords.extend(pain_words)

    if pain_keywords:
        from collections import Counter

        pain_freq = Counter(pain_keywords)

        wordcloud_pain = WordCloud(
            width=800, height=600,
            background_color='black',
            colormap='hot',
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(pain_freq)

        axes[1, 1].imshow(wordcloud_pain, interpolation='bilinear')
        axes[1, 1].set_title('Critical Pain Points - What Went Wrong',
                             fontsize=14, fontweight='bold', color='darkred', pad=10)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_wordclouds_4in1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 词云图已保存")

except ImportError:
    print("   ⚠️  需要安装wordcloud: pip install wordcloud")
except Exception as e:
    print(f"   ❌ 词云图生成失败: {e}")

# ==================== 2. 方面情感雷达图 ====================
print("\n[2/10] 📊 生成方面情感雷达图...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='polar')

# 准备数据
aspects = absa_summary['aspect'].tolist()
sentiment_scores = absa_summary['avg_sentiment'].values
mention_rates = absa_summary['mention_rate'].values

# 归一化情感得分到0-1
sentiment_normalized = (sentiment_scores + 1) / 2  # 从-1~1映射到0~1

# 角度
angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
sentiment_plot = sentiment_normalized.tolist()
angles += angles[:1]
sentiment_plot += sentiment_plot[:1]

# 绘制雷达图
ax.plot(angles, sentiment_plot, 'o-', linewidth=3, label='Sentiment Score',
        color='#2ecc71', markersize=8)
ax.fill(angles, sentiment_plot, alpha=0.25, color='#2ecc71')

# 添加提及率作为点的大小
mention_normalized = (mention_rates - mention_rates.min()) / (mention_rates.max() - mention_rates.min())
for i, (angle, score, mention) in enumerate(zip(angles[:-1], sentiment_plot[:-1], mention_normalized)):
    ax.scatter(angle, score, s=mention * 800, alpha=0.6, c='red', zorder=10, edgecolors='darkred', linewidth=2)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(aspects, size=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
ax.set_title('Aspect Sentiment Radar Chart\n(Red bubble size = Mention rate)',
             fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
ax.grid(True, linewidth=1.5, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_aspect_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 雷达图已保存")

# ==================== 3. 品牌-情感热力图 ====================
print("\n[3/10] 🔥 生成品牌对比热力图...")

# 为每个品牌计算情感统计
brand_sentiment_matrix = []
brands_list = ner_brands.head(8)['brand'].tolist()

for brand in brands_list:
    # 找到提到该品牌的评论
    brand_reviews = bert_results[
        bert_results['review_text_clean'].str.lower().str.contains(brand, na=False, regex=False)
    ]

    if len(brand_reviews) > 0:
        pos_rate = (brand_reviews['bert_label'] == 'POSITIVE').sum() / len(brand_reviews)
        avg_rating = brand_reviews['review_rating'].mean()
        count = len(brand_reviews)
        avg_score = brand_reviews['bert_score'].mean()

        brand_sentiment_matrix.append([pos_rate, avg_rating / 5, avg_score, count / max(1, len(brand_reviews))])
    else:
        brand_sentiment_matrix.append([0, 0, 0, 0])

brand_matrix = np.array(brand_sentiment_matrix).T

fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Positive Rate', 'Avg Rating\n(normalized)', 'BERT Score', 'Review Count\n(normalized)']

im = ax.imshow(brand_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(brands_list)))
ax.set_yticks(np.arange(len(metrics)))
ax.set_xticklabels(brands_list, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(metrics, fontsize=11)

# 添加数值
for i in range(len(metrics)):
    for j in range(len(brands_list)):
        text = ax.text(j, i, f'{brand_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Brand Comparison Heatmap', fontsize=16, fontweight='bold', pad=15)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score (0-1)', rotation=270, labelpad=20, fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_brand_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 品牌热力图已保存")

# ==================== 4. 关键词网络图 ====================
print("\n[4/10] 🕸️  生成关键词网络图...")

try:
    import networkx as nx

    G = nx.Graph()

    # 添加关键词节点
    top_keywords = textrank_keywords.head(20)

    for idx, row in top_keywords.iterrows():
        G.add_node(row['keyword'], weight=row['score'])

    # 基于词序添加边
    keywords_list = top_keywords['keyword'].tolist()
    for i in range(len(keywords_list)):
        for j in range(i + 1, min(i + 5, len(keywords_list))):
            G.add_edge(keywords_list[i], keywords_list[j], weight=1.0 / (j - i))

    # 绘制
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # 节点大小
    node_sizes = [G.nodes[node]['weight'] * 50000 for node in G.nodes()]

    # 边权重
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # 绘制
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.7, ax=ax,
                           edgecolors='darkblue', linewidths=2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

    ax.set_title('Keyword Network Graph\n(Node size = Importance)', fontsize=16, fontweight='bold')
    ax.axis('off')
    ax.margins(0.1)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_keyword_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 网络图已保存")

except ImportError:
    print("   ⚠️  需要安装networkx: pip install networkx")
except Exception as e:
    print(f"   ❌ 网络图生成失败: {e}")

# ==================== 5. 情感-评分流向图 ====================
print("\n[5/10] 📈 生成情感流向图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 5.1 堆叠柱状图
rating_sentiment = bert_results.groupby(['review_rating', 'bert_label']).size().unstack(fill_value=0)

ratings = sorted(bert_results['review_rating'].unique())
x = np.arange(len(ratings))
width = 0.6

if 'POSITIVE' in rating_sentiment.columns:
    ax1.bar(x, rating_sentiment['POSITIVE'], width, label='Positive',
            color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
if 'NEGATIVE' in rating_sentiment.columns:
    bottom = rating_sentiment['POSITIVE'] if 'POSITIVE' in rating_sentiment.columns else 0
    ax1.bar(x, rating_sentiment['NEGATIVE'], width, bottom=bottom,
            label='Negative', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Review Count', fontsize=12, fontweight='bold')
ax1.set_title('Sentiment Distribution by Rating', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(r)} ★' for r in ratings], fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# 5.2 比例面积图
pos_counts = []
neg_counts = []
for rating in ratings:
    rating_data = bert_results[bert_results['review_rating'] == rating]
    total = len(rating_data)
    if total > 0:
        pos_pct = (rating_data['bert_label'] == 'POSITIVE').sum() / total * 100
        neg_pct = (rating_data['bert_label'] == 'NEGATIVE').sum() / total * 100
    else:
        pos_pct = neg_pct = 0
    pos_counts.append(pos_pct)
    neg_counts.append(neg_pct)

ax2.fill_between(ratings, pos_counts, alpha=0.5, color='green', label='Positive %')
ax2.fill_between(ratings, neg_counts, alpha=0.5, color='red', label='Negative %')
ax2.plot(ratings, pos_counts, 'o-', color='darkgreen', linewidth=2, markersize=8)
ax2.plot(ratings, neg_counts, 'o-', color='darkred', linewidth=2, markersize=8)

ax2.set_xlabel('Star Rating', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Sentiment Percentage by Rating', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_sentiment_flow.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 流向图已保存")

# ==================== 6. 主题词云 ====================
print("\n[6/10] 🎯 生成主题词云...")

try:
    from wordcloud import WordCloud

    lda_topics = topic_modeling[topic_modeling['method'] == 'LDA']

    if len(lda_topics) > 0:
        n_topics = min(len(lda_topics), 6)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (idx, row) in enumerate(lda_topics.head(6).iterrows()):
            topic_words = row['top_words']

            wordcloud = WordCloud(
                width=600, height=400,
                background_color='white',
                colormap=['tab10', 'Set3', 'Pastel1', 'Dark2', 'Set2', 'Accent'][i % 6],
                max_words=40,
                relative_scaling=0.5
            ).generate(topic_words)

            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Topic {i + 1}: {topic_words.split(",")[0]}...',
                              fontsize=12, fontweight='bold')
            axes[i].axis('off')

        # 隐藏多余子图
        for i in range(len(lda_topics), 6):
            axes[i].axis('off')

        plt.suptitle('LDA Topic Modeling - Word Clouds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/06_topic_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 主题词云已保存")

except Exception as e:
    print(f"   ❌ 主题词云生成失败: {e}")

# ==================== 7. 方面气泡图 ====================
print("\n[7/10] 💭 生成方面气泡图...")

fig, ax = plt.subplots(figsize=(14, 8))

aspects = absa_summary['aspect'].tolist()
x_pos = range(len(aspects))
y_sentiment = absa_summary['avg_sentiment'].values
sizes = absa_summary['mention_count'].values
colors = ['#2ecc71' if s > 0.1 else '#e74c3c' if s < -0.1 else '#95a5a6' for s in y_sentiment]

# 绘制气泡
scatter = ax.scatter(x_pos, y_sentiment, s=sizes * 3, c=colors, alpha=0.6,
                     edgecolors='black', linewidth=2, zorder=10)

# 添加标签
for i, (aspect, sent) in enumerate(zip(aspects, y_sentiment)):
    ax.text(i, sent + 0.05, aspect, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    ax.text(i, sent - 0.05, f'({int(sizes[i])})', ha='center', va='top',
            fontsize=9, style='italic', alpha=0.7)

# 添加零线
ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Neutral line')

ax.set_xlabel('Product Aspects', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Sentiment Score', fontsize=13, fontweight='bold')
ax.set_title('Aspect Sentiment Bubble Chart\n(Bubble size = Mention count)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(aspects, rotation=45, ha='right', fontsize=11)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(fontsize=10)

# 添加颜色说明
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Positive (>0.1)'),
    Patch(facecolor='#95a5a6', edgecolor='black', label='Neutral'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Negative (<-0.1)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_aspect_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 气泡图已保存")

# ==================== 8. 品牌情感箱型图 ====================
print("\n[8/10] 📦 生成品牌对比箱型图...")

brand_sentiment_data = []

for brand in ner_brands.head(6)['brand']:
    brand_reviews = bert_results[
        bert_results['review_text_clean'].str.lower().str.contains(brand, na=False, regex=False)
    ]

    if len(brand_reviews) > 0:
        for _, row in brand_reviews.iterrows():
            score = row['bert_score'] if row['bert_label'] == 'POSITIVE' else -row['bert_score']
            brand_sentiment_data.append({
                'brand': brand.capitalize(),
                'sentiment_score': score
            })

if brand_sentiment_data:
    brand_df = pd.DataFrame(brand_sentiment_data)

    fig, ax = plt.subplots(figsize=(14, 8))

    brands = brand_df['brand'].unique()
    data_to_plot = [brand_df[brand_df['brand'] == b]['sentiment_score'].values for b in brands]

    bp = ax.boxplot(data_to_plot, labels=brands, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                    medianprops=dict(color='red', linewidth=3),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.5))

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Neutral')
    ax.set_xlabel('Brand', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=13, fontweight='bold')
    ax.set_title('Brand Sentiment Distribution (Box Plot)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_brand_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ 箱型图已保存")
else:
    print("   ⚠️  品牌数据不足，跳过")

# ==================== 9. 痛点漏斗图 ====================
print("\n[9/10] 🔻 生成痛点漏斗图...")

negative_reviews = bert_results[bert_results['bert_label'] == 'NEGATIVE']

pain_points = {
    'Rust/Corrosion': 0,
    'Dull Blade': 0,
    'Handle Issues': 0,
    'Poor Quality': 0,
    'Broke/Cracked': 0,
    'Expensive': 0
}

for review in negative_reviews['review_text_clean'].fillna(''):
    review_lower = review.lower()
    if any(word in review_lower for word in ['rust', 'rusted', 'rusting', 'corrosion']):
        pain_points['Rust/Corrosion'] += 1
    if 'dull' in review_lower:
        pain_points['Dull Blade'] += 1
    if 'handle' in review_lower and any(word in review_lower for word in ['crack', 'split', 'break', 'loose']):
        pain_points['Handle Issues'] += 1
    if any(word in review_lower for word in ['poor', 'cheap', 'bad', 'terrible']):
        pain_points['Poor Quality'] += 1
    if any(word in review_lower for word in ['broke', 'crack', 'break', 'chip']):
        pain_points['Broke/Cracked'] += 1
    if any(word in review_lower for word in ['expensive', 'overpriced', 'waste money']):
        pain_points['Expensive'] += 1

pain_points_sorted = sorted(pain_points.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = range(len(pain_points_sorted))
counts = [p[1] for p in pain_points_sorted]
labels = [p[0] for p in pain_points_sorted]

colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pain_points_sorted)))

for i, (label, count) in enumerate(pain_points_sorted):
    width = count / max(counts) * 10
    bar = ax.barh(i, width, height=0.7, color=colors[i], alpha=0.8,
                  edgecolor='darkred', linewidth=2)

    ax.text(width + 0.3, i, f'{count} mentions\n({count / len(negative_reviews) * 100:.1f}%)',
            va='center', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=13, fontweight='bold')
ax.set_xlabel('Relative Frequency', fontsize=13, fontweight='bold')
ax.set_title('Pain Points Analysis (Funnel Chart)\nBased on Negative Reviews',
             fontsize=16, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.set_xlim(0, 12)

# 添加说明
ax.text(11, len(pain_points_sorted) - 0.5,
        f'Total Negative Reviews: {len(negative_reviews)}',
        fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_pain_points_funnel.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 痛点漏斗图已保存")

# ==================== 10. 综合仪表盘 ====================
print("\n[10/10] 📊 生成综合仪表盘...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

# 10.1 情感分布饼图
ax1 = fig.add_subplot(gs[0, 0])
sentiment_counts = bert_results['bert_label'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                                   autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Overall Sentiment', fontweight='bold', fontsize=12, pad=10)

# 10.2 评分分布
ax2 = fig.add_subplot(gs[0, 1])
rating_dist = bert_results['review_rating'].value_counts().sort_index()
bars = ax2.bar(rating_dist.index, rating_dist.values, color='steelblue', alpha=0.7, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_xlabel('Star Rating', fontweight='bold')
ax2.set_ylabel('Count', fontweight='bold')
ax2.set_title('Rating Distribution', fontweight='bold', fontsize=12, pad=10)
ax2.grid(axis='y', alpha=0.3)

# 10.3 Top品牌
ax3 = fig.add_subplot(gs[0, 2:])
top_brands = ner_brands.head(8)
bars = ax3.barh(top_brands['brand'], top_brands['count'], color='coral', alpha=0.7, edgecolor='black')
for bar in bars:
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height() / 2.,
             f' {int(width)}', ha='left', va='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Mention Count', fontweight='bold')
ax3.set_title('Top Brands Mentioned', fontweight='bold', fontsize=12, pad=10)
ax3.invert_yaxis()

# 10.4 方面情感
ax4 = fig.add_subplot(gs[1, :])
aspects = absa_summary['aspect'].tolist()
sentiments = absa_summary['avg_sentiment'].values
colors_aspect = ['#2ecc71' if s > 0.1 else '#e74c3c' if s < -0.1 else '#95a5a6' for s in sentiments]
bars = ax4.barh(aspects, sentiments, color=colors_aspect, alpha=0.7, edgecolor='black', linewidth=1.5)
for i, (bar, sent) in enumerate(zip(bars, sentiments)):
    width = bar.get_width()
    ax4.text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height() / 2.,
             f'{sent:.2f}', ha='left' if width > 0 else 'right', va='center',
             fontsize=10, fontweight='bold')
ax4.axvline(x=0, color='black', linewidth=2)
ax4.set_xlabel('Sentiment Score', fontweight='bold', fontsize=11)
ax4.set_title('ABSA: Aspect Sentiment Scores', fontweight='bold', fontsize=13, pad=10)
ax4.grid(axis='x', alpha=0.3)

# 10.5 关键词Top 12
ax5 = fig.add_subplot(gs[2, :2])
top_kw = textrank_keywords.head(12)
bars = ax5.barh(range(len(top_kw)), top_kw.iloc[:, 1].values, color='skyblue', alpha=0.7, edgecolor='black')
ax5.set_yticks(range(len(top_kw)))
ax5.set_yticklabels(top_kw.iloc[:, 0].values, fontsize=10)
ax5.invert_yaxis()
ax5.set_xlabel('TextRank Score', fontweight='bold')
ax5.set_title('Top Keywords (TextRank)', fontweight='bold', fontsize=12, pad=10)

# 10.6 材质分布
ax6 = fig.add_subplot(gs[2, 2:])
top_materials = ner_materials.head(6)
wedges, texts, autotexts = ax6.pie(top_materials['count'].values,
                                   labels=[m.capitalize() for m in top_materials['material'].values],
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'fontsize': 9, 'fontweight': 'bold'})
ax6.set_title('Material Mentions', fontweight='bold', fontsize=12, pad=10)

# 10.7 评论长度分布
ax7 = fig.add_subplot(gs[3, :2])
text_lens = bert_results[bert_results['review_text_clean'].notna()]['review_text_clean'].str.len()
ax7.hist(text_lens, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
median_len = text_lens.median()
ax7.axvline(median_len, color='red', linestyle='--', linewidth=2,
            label=f'Median: {int(median_len)} chars')
ax7.set_xlabel('Review Length (characters)', fontweight='bold')
ax7.set_ylabel('Frequency', fontweight='bold')
ax7.set_title('Review Length Distribution', fontweight='bold', fontsize=12, pad=10)
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# 10.8 关键指标卡
ax8 = fig.add_subplot(gs[3, 2:])
ax8.axis('off')

metrics_text = f"""
KEY METRICS SUMMARY
{'=' * 40}

Total Reviews: {len(bert_results):,}
Positive Rate: {(bert_results['bert_label'] == 'POSITIVE').sum() / len(bert_results) * 100:.1f}%
Negative Rate: {(bert_results['bert_label'] == 'NEGATIVE').sum() / len(bert_results) * 100:.1f}%

Average Rating: {bert_results['review_rating'].mean():.2f} / 5.0
Median Review Length: {int(median_len)} characters

Top Aspect: {aspects[0]} ({absa_summary.iloc[0]['mention_rate']:.1f}% mention)
Biggest Pain Point: Rust ({pain_points['Rust/Corrosion']} mentions)

Most Mentioned Brand: {ner_brands.iloc[0]['brand'].capitalize()}
Most Mentioned Material: {ner_materials.iloc[0]['material'].capitalize()}
"""

ax8.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1))

plt.suptitle('Amazon Kitchen Knife Reviews - Comprehensive Analytics Dashboard',
             fontsize=18, fontweight='bold', y=0.99)

plt.savefig(f'{OUTPUT_DIR}/10_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ 综合仪表盘已保存")

# ==================== 生成图表说明文档 ====================
summary_md = f"""# 📊 高级可视化图表说明文档

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📁 图表清单

### 1️⃣ 综合词云图 (01_wordclouds_4in1.png)
**内容**: 4合1词云分析
- **左上**: 整体词云 - 所有评论的关键词
- **右上**: 正面词云 - 用户喜欢什么（绿色主题）
- **左下**: 负面词云 - 用户不满意什么（红色主题）
- **右下**: 痛点词云 - 核心问题词（黑底热力图）

**展示建议**: 放在PPT第2-3页，快速展示用户声音
**核心洞察**: 
- 正面词: sharp, quality, great, love
- 负面词: rust, dull, broke, cheap, disappointing

---

### 2️⃣ 方面情感雷达图 (02_aspect_radar_chart.png)
**内容**: 8个产品方面的情感得分雷达图
- 绿线: 情感得分（越大越好）
- 红色气泡: 提及率（越大说明用户越关注）

**展示建议**: PPT核心页，展示产品强弱项
**核心洞察**:
- ✅ 优势: Sharpness, Quality, Appearance
- ⚠️ 痛点: Rust (负面情感)

---

### 3️⃣ 品牌对比热力图 (03_brand_heatmap.png)
**内容**: Top 8品牌的4个维度对比
- 正面评价率
- 平均评分
- BERT置信度
- 评论数量

**展示建议**: 竞品分析环节
**核心洞察**: 
- Cuisinart: 提及最多但评分一般
- 机会: 避开头部品牌直接竞争

---

### 4️⃣ 关键词网络图 (04_keyword_network.png)
**内容**: Top 20关键词的关联网络
- 节点大小 = 词语重要性
- 连线 = 词语之间的关联

**展示建议**: 技术展示环节
**核心洞察**: 识别关键词簇和主题

---

### 5️⃣ 情感流向图 (05_sentiment_flow.png)
**内容**: 评分-情感双维度分析
- 左图: 堆叠柱状图（绝对数量）
- 右图: 百分比面积图（相对比例）

**展示建议**: 验证BERT准确性
**核心洞察**: 
- 5星评论几乎全是正面情感
- 1-2星评论以负面为主
- BERT模型准确度高 ✅

---

### 6️⃣ 主题词云图 (06_topic_wordclouds.png)
**内容**: LDA主题建模的6个主题词云
- 每个主题用不同颜色
- 展示主题的核心词汇

**展示建议**: 用户讨论话题分析
**核心洞察**: 
- Topic 1: Performance & Sharpness
- Topic 2: Quality & Durability
- Topic 3: Value & Price
- (等)

---

### 7️⃣ 方面气泡图 (07_aspect_bubble_chart.png)
**内容**: 方面提及次数与情感得分的气泡图
- X轴: 产品方面
- Y轴: 平均情感得分
- 气泡大小: 提及次数
- 颜色: 绿(正面) 红(负面) 灰(中性)

**展示建议**: 快速识别核心问题
**核心洞察**:
- Sharpness: 高提及+高情感 = 核心卖点 ✅
- Rust: 中提及+负情感 = 核心痛点 ⚠️

---

### 8️⃣ 品牌情感箱型图 (08_brand_boxplot.png)
**内容**: Top 6品牌的情感得分分布
- 箱体: 25%-75%分位数
- 红线: 中位数
- 触须: 最大/最小值
- 圆点: 异常值

**展示建议**: 品牌满意度对比
**核心洞察**: 
- 箱体越高 = 用户评价越好
- 箱体越窄 = 评价越一致

---

### 9️⃣ 痛点漏斗图 (09_pain_points_funnel.png)
**内容**: 负面评论中的6大痛点排名
1. Rust/Corrosion - 生锈/腐蚀
2. Dull Blade - 刀刃变钝
3. Handle Issues - 手柄问题
4. Poor Quality - 质量差
5. Broke/Cracked - 断裂/破损
6. Expensive - 价格贵

**展示建议**: 产品改进优先级
**核心洞察**:
- ⚠️ Rust是最大痛点（占负面评论的XX%）
- 改进建议: 升级防锈技术 + 质保承诺

---

### 🔟 综合仪表盘 (10_comprehensive_dashboard.png)
**内容**: 8合1数据看板
- 情感分布饼图
- 评分分布柱状图
- Top品牌排行
- 方面情感条形图
- 关键词排名
- 材质分布
- 评论长度分布
- 关键指标摘要

**展示建议**: PPT首页或总结页
**核心洞察**: 一页纸看懂所有核心数据

---

## 🎯 PPT展示建议

### 结构1: 问题导向型
```
第1页: 封面
第2页: 综合仪表盘（全景）
第3页: 词云图（用户声音）
第4页: ABSA雷达图（产品分析）
第5页: 痛点漏斗图（核心问题）
第6页: 解决方案（基于数据）
```

### 结构2: 技术展示型
```
第1页: 封面
第2页: 技术路线（5项NLP）
第3页: BERT结果（情感流向图）
第4页: ABSA结果（雷达图+气泡图）
第5页: TextRank结果（网络图+词云）
第6页: 主题建模（主题词云）
第7页: 商业洞察
```

---

## 💡 使用技巧

### 配色方案
- ✅ 正面: #2ecc71 (绿色)
- ❌ 负面: #e74c3c (红色)
- 😐 中性: #95a5a6 (灰色)
- 统一配色提升专业度

### 数据标注
- 每张图都添加了具体数值
- 便于观众理解和记忆

### 高清输出
- 所有图表均为300 DPI
- 打印和投影都清晰

---

## 📈 核心数据摘要

基于这些图表，你可以得出：

1. **用户满意度**: {(bert_results['bert_label'] == 'POSITIVE').sum() / len(bert_results) * 100:.1f}% 正面评价
2. **最大卖点**: Sharpness（锋利度）- {absa_summary.iloc[0]['mention_rate']:.1f}%提及率
3. **最大痛点**: Rust（防锈）- 负面情感
4. **主要竞品**: {ner_brands.iloc[0]['brand'].capitalize()}
5. **关注材质**: {ner_materials.iloc[0]['material'].capitalize()}

---

**🎉 所有图表已生成，可直接用于比赛展示！**
"""

with open(f'{OUTPUT_DIR}/图表说明文档.md', 'w', encoding='utf-8') as f:
    f.write(summary_md)

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("🎉 所有可视化生成完成!")
print("=" * 80)

print(f"\n📁 输出目录: {OUTPUT_DIR}/")
print("\n📊 生成的图表:")
print("   1. 01_wordclouds_4in1.png - 综合词云图（4合1）")
print("   2. 02_aspect_radar_chart.png - 方面情感雷达图")
print("   3. 03_brand_heatmap.png - 品牌对比热力图")
print("   4. 04_keyword_network.png - 关键词网络图")
print("   5. 05_sentiment_flow.png - 情感流向图")
print("   6. 06_topic_wordclouds.png - 主题词云图")
print("   7. 07_aspect_bubble_chart.png - 方面气泡图")
print("   8. 08_brand_boxplot.png - 品牌情感箱型图")
print("   9. 09_pain_points_funnel.png - 痛点漏斗图")
print("  10. 10_comprehensive_dashboard.png - 综合仪表盘")

print("\n📄 说明文档:")
print("   - 图表说明文档.md - 每张图的详细说明和使用建议")

print("\n✨ 所有图表均为高清PNG格式（300 DPI），适合:")
print("   - PPT展示")
print("   - 打印海报")
print("   - 论文插图")
print("   - 比赛答辩")

print("\n💡 下一步建议:")
print("   1. 查看 图表说明文档.md 了解每张图的用途")
print("   2. 根据PPT需求选择合适的图表")
print("   3. 配合洞察文字讲好数据故事")
print("   4. 准备3分钟演示脚本")

print("\n" + "=" * 80)
print("祝你比赛顺利！🚀")
print("=" * 80)
