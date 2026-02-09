"""
亚马逊厨刀评论NLP完整分析系统
包含: BERT情感分析、ABSA、TextRank、NER、主题建模、可视化

使用方法:
1. 确保数据文件在项目根目录
2. 安装依赖: pip install -r requirements.txt
3. 运行: python nlp_analysis_complete.py

作者: [Your Name]
日期: 2026-01-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
from collections import Counter
import re

warnings.filterwarnings('ignore')

# 设置中文和样式
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

# ==================== 配置 ====================
class Config:
    """配置参数"""
    # 数据文件路径（在项目根目录）
    REVIEWS_FILE = 'reviews_cleaned.csv'
    FACT_REVIEWS_FILE = 'fact_review_enriched.csv'
    PRODUCTS_FILE = 'products_clean.csv'
    
    # 输出目录
    OUTPUT_DIR = 'nlp_results'
    VIZ_DIR = 'nlp_results/visualizations'
    DATA_DIR = 'nlp_results/data'
    
    # 分析参数
    MIN_REVIEW_LENGTH = 20  # 最小评论长度
    N_TOPICS = 5  # 主题数量
    TOP_KEYWORDS = 20  # 关键词数量
    
    # 厨刀方面定义
    ASPECTS = {
        'sharpness': ['sharp', 'blade', 'edge', 'dull', 'cutting', 'razor', 'keen'],
        'quality': ['quality', 'made', 'construction', 'build', 'material', 'craftsmanship'],
        'durability': ['durable', 'last', 'lasting', 'sturdy', 'strong', 'break', 'broke', 'chip'],
        'handle': ['handle', 'grip', 'comfortable', 'ergonomic', 'hold', 'hand'],
        'rust': ['rust', 'rusted', 'corrosion', 'stain', 'stainless', 'oxidation'],
        'balance': ['balance', 'balanced', 'weight', 'heavy', 'light'],
        'value': ['price', 'value', 'money', 'worth', 'expensive', 'cheap', 'cost'],
        'appearance': ['look', 'beautiful', 'pretty', 'appearance', 'design', 'aesthetic']
    }

# 创建输出目录
def setup_directories():
    """创建输出目录结构"""
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(Config.VIZ_DIR).mkdir(exist_ok=True)
    Path(Config.DATA_DIR).mkdir(exist_ok=True)
    print(f"✓ 输出目录已创建: {Config.OUTPUT_DIR}")

# ==================== 数据加载 ====================
def load_data():
    """加载数据"""
    print("\n" + "="*80)
    print("【第一步：数据加载】")
    print("="*80)
    
    try:
        reviews = pd.read_csv(Config.REVIEWS_FILE, encoding='utf-8-sig')
        fact_reviews = pd.read_csv(Config.FACT_REVIEWS_FILE, encoding='utf-8-sig')
        products = pd.read_csv(Config.PRODUCTS_FILE, encoding='utf-8-sig')
        
        print(f"✓ 评论数据: {len(reviews):,} 条")
        print(f"✓ 增强评论: {len(fact_reviews):,} 条")
        print(f"✓ 产品数据: {len(products):,} 个")
        
        # 只保留有文本的评论
        text_reviews = reviews[reviews['has_text'] == 1].copy()
        print(f"✓ 有效文本评论: {len(text_reviews):,} 条")
        
        return text_reviews, fact_reviews, products
        
    except FileNotFoundError as e:
        print(f"✗ 错误: 找不到数据文件 - {e}")
        print("请确保以下文件在项目根目录:")
        print(f"  - {Config.REVIEWS_FILE}")
        print(f"  - {Config.FACT_REVIEWS_FILE}")
        print(f"  - {Config.PRODUCTS_FILE}")
        raise

# ==================== 1. BERT情感分析 ====================
def bert_sentiment_analysis(reviews_df):
    """使用BERT进行情感分析"""
    print("\n" + "="*80)
    print("【第二步：BERT情感分析】")
    print("="*80)
    
    try:
        from transformers import pipeline
        import torch
        
        print("正在加载BERT模型（首次运行需要下载，约500MB）...")
        
        # 使用CPU友好的DistilBERT
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # 使用CPU
        )
        
        print("✓ 模型加载成功")
        
        # 批量处理
        texts = reviews_df['review_text_clean'].fillna('').tolist()
        batch_size = 32
        results = []
        
        print(f"开始分析 {len(texts)} 条评论...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # 截断长文本
            batch = [text[:512] for text in batch]
            batch_results = sentiment_analyzer(batch)
            results.extend(batch_results)
            
            if (i + batch_size) % 500 == 0:
                print(f"  进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        # 整理结果
        reviews_df['bert_label'] = [r['label'] for r in results]
        reviews_df['bert_score'] = [r['score'] for r in results]
        
        # 统计
        label_dist = reviews_df['bert_label'].value_counts()
        print(f"\n✓ BERT情感分析完成!")
        print(f"\n情感分布:")
        for label, count in label_dist.items():
            pct = count / len(reviews_df) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # 保存结果
        output_file = f"{Config.DATA_DIR}/bert_sentiment_results.csv"
        reviews_df[['review_id', 'review_text_clean', 'review_rating', 
                    'bert_label', 'bert_score']].to_csv(output_file, index=False)
        print(f"\n✓ 结果已保存: {output_file}")
        
        return reviews_df
        
    except Exception as e:
        print(f"✗ BERT分析失败: {e}")
        print("使用备用方案：基于规则的情感分析")
        
        # 备用方案：规则分析
        positive_words = {'excellent', 'great', 'amazing', 'perfect', 'love', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate'}
        
        def simple_sentiment(text):
            text = str(text).lower()
            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)
            return 'POSITIVE' if pos > neg else ('NEGATIVE' if neg > pos else 'NEUTRAL')
        
        reviews_df['bert_label'] = reviews_df['review_text_clean'].apply(simple_sentiment)
        reviews_df['bert_score'] = 0.5
        
        return reviews_df

# ==================== 2. ABSA方面级情感分析 ====================
def absa_analysis(reviews_df):
    """方面级情感分析"""
    print("\n" + "="*80)
    print("【第三步：ABSA方面级情感分析】")
    print("="*80)
    
    print(f"分析 {len(Config.ASPECTS)} 个产品方面...")
    
    aspect_data = []
    
    for idx, row in reviews_df.iterrows():
        text = str(row['review_text_clean']).lower()
        rating = row['review_rating']
        bert_label = row.get('bert_label', 'NEUTRAL')
        
        for aspect_name, keywords in Config.ASPECTS.items():
            # 检查是否提到该方面
            mentioned = any(kw in text for kw in keywords)
            
            if mentioned:
                # 提取相关句子
                sentences = text.split('.')
                relevant_sentences = [s for s in sentences if any(kw in s for kw in keywords)]
                
                if relevant_sentences:
                    # 简单情感判断：基于星级和BERT结果
                    if rating >= 4 and bert_label == 'POSITIVE':
                        sentiment = 'positive'
                        score = 1.0
                    elif rating <= 2 or bert_label == 'NEGATIVE':
                        sentiment = 'negative'
                        score = -1.0
                    else:
                        sentiment = 'neutral'
                        score = 0.0
                    
                    aspect_data.append({
                        'review_id': row['review_id'],
                        'aspect': aspect_name,
                        'sentiment': sentiment,
                        'score': score,
                        'rating': rating,
                        'sample_text': relevant_sentences[0][:100]
                    })
        
        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{len(reviews_df)}")
    
    aspect_df = pd.DataFrame(aspect_data)
    
    # 统计每个方面
    aspect_stats = aspect_df.groupby('aspect').agg({
        'score': ['mean', 'count'],
        'rating': 'mean'
    }).round(3)
    
    aspect_stats.columns = ['avg_sentiment', 'mention_count', 'avg_rating']
    aspect_stats['mention_rate'] = (aspect_stats['mention_count'] / len(reviews_df) * 100).round(1)
    aspect_stats = aspect_stats.sort_values('mention_count', ascending=False)
    
    print(f"\n✓ 方面级分析完成!")
    print(f"\n{aspect_stats.to_string()}")
    
    # 保存结果
    aspect_df.to_csv(f"{Config.DATA_DIR}/absa_detailed.csv", index=False)
    aspect_stats.to_csv(f"{Config.DATA_DIR}/absa_summary.csv")
    print(f"\n✓ 结果已保存: {Config.DATA_DIR}/absa_*.csv")
    
    return aspect_df, aspect_stats

# ==================== 3. TextRank关键词提取 ====================
def textrank_keywords(reviews_df):
    """使用TextRank提取关键词"""
    print("\n" + "="*80)
    print("【第四步：TextRank关键词提取】")
    print("="*80)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import networkx as nx
        
        # 合并所有评论
        all_text = ' '.join(reviews_df['review_text_clean'].fillna('').tolist())
        
        # 分词
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        
        # 停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been'
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        print(f"处理 {len(words):,} 个词...")
        
        # 构建共现图
        graph = nx.Graph()
        window_size = 5
        
        for i in range(len(words) - window_size):
            for j in range(i + 1, i + window_size):
                if words[i] != words[j]:
                    if graph.has_edge(words[i], words[j]):
                        graph[words[i]][words[j]]['weight'] += 1
                    else:
                        graph.add_edge(words[i], words[j], weight=1)
        
        print(f"图节点数: {len(graph.nodes())}")
        
        # PageRank
        scores = nx.pagerank(graph, weight='weight')
        
        # 排序
        keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:Config.TOP_KEYWORDS]
        
        print(f"\n✓ TextRank关键词提取完成!")
        print(f"\nTop {Config.TOP_KEYWORDS} 关键词:")
        for i, (word, score) in enumerate(keywords, 1):
            print(f"  {i:2}. {word:15} - {score:.6f}")
        
        # 保存结果
        keywords_df = pd.DataFrame(keywords, columns=['keyword', 'score'])
        keywords_df.to_csv(f"{Config.DATA_DIR}/textrank_keywords.csv", index=False)
        print(f"\n✓ 结果已保存: {Config.DATA_DIR}/textrank_keywords.csv")
        
        return keywords_df
        
    except Exception as e:
        print(f"✗ TextRank失败: {e}")
        print("使用备用方案：简单词频统计")
        
        from collections import Counter
        all_text = ' '.join(reviews_df['review_text_clean'].fillna('').tolist())
        words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
        word_freq = Counter(words).most_common(Config.TOP_KEYWORDS)
        
        keywords_df = pd.DataFrame(word_freq, columns=['keyword', 'frequency'])
        return keywords_df

# ==================== 4. 简单NER ====================
def simple_ner(reviews_df):
    """简单的命名实体识别（品牌和材质）"""
    print("\n" + "="*80)
    print("【第五步：命名实体识别（NER）】")
    print("="*80)
    
    # 常见品牌和材质
    brands = [
        'wusthof', 'shun', 'victorinox', 'zwilling', 'henckels', 'global',
        'miyabi', 'dalstrong', 'cuisinart', 'farberware', 'imarku', 'paudin',
        'hoshanho', 'mercer', 'chicago cutlery', 'j.a. henckels', 'cutco'
    ]
    
    materials = [
        'steel', 'stainless', 'carbon', 'damascus', 'ceramic', 'titanium',
        'german steel', 'japanese steel', 'high carbon', 'vg-10', 'vg-max',
        'aus-8', 'x50crmov15'
    ]
    
    print("识别品牌和材质...")
    
    all_brands = []
    all_materials = []
    
    for text in reviews_df['review_text_clean'].fillna(''):
        text_lower = text.lower()
        all_brands.extend([b for b in brands if b in text_lower])
        all_materials.extend([m for m in materials if m in text_lower])
    
    brand_counts = Counter(all_brands)
    material_counts = Counter(all_materials)
    
    print(f"\n✓ NER完成!")
    print(f"\n品牌提及 Top 10:")
    for i, (brand, count) in enumerate(brand_counts.most_common(10), 1):
        print(f"  {i:2}. {brand:20} - {count:3} 次")
    
    print(f"\n材质提及 Top 10:")
    for i, (material, count) in enumerate(material_counts.most_common(10), 1):
        print(f"  {i:2}. {material:20} - {count:3} 次")
    
    # 保存结果
    brand_df = pd.DataFrame(brand_counts.most_common(), columns=['brand', 'count'])
    material_df = pd.DataFrame(material_counts.most_common(), columns=['material', 'count'])
    
    brand_df.to_csv(f"{Config.DATA_DIR}/ner_brands.csv", index=False)
    material_df.to_csv(f"{Config.DATA_DIR}/ner_materials.csv", index=False)
    print(f"\n✓ 结果已保存: {Config.DATA_DIR}/ner_*.csv")
    
    return brand_df, material_df

# ==================== 5. 主题建模 ====================
def topic_modeling(reviews_df):
    """LDA和NMF主题建模"""
    print("\n" + "="*80)
    print("【第六步：主题建模（LDA + NMF）】")
    print("="*80)
    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    
    # 只使用长评论
    long_reviews = reviews_df[reviews_df['text_len'] > Config.MIN_REVIEW_LENGTH]
    texts = long_reviews['review_text_clean'].fillna('').tolist()
    
    print(f"使用 {len(texts):,} 条长评论进行主题建模...")
    
    # ===== LDA =====
    print("\n执行LDA主题建模...")
    vectorizer_lda = CountVectorizer(
        max_features=500,
        stop_words='english',
        min_df=3,
        max_df=0.7
    )
    
    doc_term_matrix = vectorizer_lda.fit_transform(texts)
    
    lda = LatentDirichletAllocation(
        n_components=Config.N_TOPICS,
        random_state=42,
        max_iter=20
    )
    
    lda.fit(doc_term_matrix)
    
    # 提取LDA主题
    feature_names = vectorizer_lda.get_feature_names_out()
    lda_topics = []
    
    print(f"\nLDA发现的 {Config.N_TOPICS} 个主题:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        lda_topics.append({
            'topic_id': topic_idx,
            'method': 'LDA',
            'top_words': ', '.join(top_words[:8])
        })
        print(f"  主题 {topic_idx + 1}: {', '.join(top_words[:8])}")
    
    # ===== NMF =====
    print("\n执行NMF主题建模...")
    vectorizer_nmf = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=3,
        max_df=0.7
    )
    
    tfidf_matrix = vectorizer_nmf.fit_transform(texts)
    
    nmf = NMF(
        n_components=Config.N_TOPICS,
        random_state=42,
        max_iter=200
    )
    
    nmf.fit(tfidf_matrix)
    
    # 提取NMF主题
    feature_names_nmf = vectorizer_nmf.get_feature_names_out()
    nmf_topics = []
    
    print(f"\nNMF发现的 {Config.N_TOPICS} 个主题:")
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names_nmf[i] for i in top_words_idx]
        nmf_topics.append({
            'topic_id': topic_idx,
            'method': 'NMF',
            'top_words': ', '.join(top_words[:8])
        })
        print(f"  主题 {topic_idx + 1}: {', '.join(top_words[:8])}")
    
    # 保存结果
    topics_df = pd.DataFrame(lda_topics + nmf_topics)
    topics_df.to_csv(f"{Config.DATA_DIR}/topic_modeling.csv", index=False)
    print(f"\n✓ 结果已保存: {Config.DATA_DIR}/topic_modeling.csv")
    
    return lda_topics, nmf_topics

# ==================== 可视化 ====================
def create_visualizations(reviews_df, aspect_stats, keywords_df, brand_df, material_df):
    """创建所有可视化图表"""
    print("\n" + "="*80)
    print("【第七步：生成可视化图表】")
    print("="*80)
    
    # 1. BERT情感分布
    print("1. BERT情感分布图...")
    plt.figure(figsize=(10, 6))
    sentiment_counts = reviews_df['bert_label'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('BERT Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.savefig(f"{Config.VIZ_DIR}/1_bert_sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. BERT情感vs星级
    print("2. 情感vs星级对比图...")
    plt.figure(figsize=(12, 6))
    sentiment_by_rating = reviews_df.groupby(['review_rating', 'bert_label']).size().unstack(fill_value=0)
    sentiment_by_rating.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'])
    plt.title('Sentiment Distribution by Star Rating', fontsize=16, fontweight='bold')
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='BERT Sentiment')
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/2_sentiment_by_rating.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ABSA方面情感热力图
    print("3. ABSA方面情感热力图...")
    plt.figure(figsize=(10, 8))
    heatmap_data = aspect_stats[['avg_sentiment', 'mention_rate']].sort_values('mention_rate', ascending=False)
    sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Score'})
    plt.title('ABSA: Aspect Sentiment Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('Metric', fontsize=12)
    plt.xlabel('Aspect', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/3_absa_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ABSA方面提及率
    print("4. ABSA方面提及率...")
    plt.figure(figsize=(12, 6))
    aspect_stats_sorted = aspect_stats.sort_values('mention_rate', ascending=True)
    colors_aspect = ['#2ecc71' if x > 0 else '#e74c3c' for x in aspect_stats_sorted['avg_sentiment']]
    plt.barh(aspect_stats_sorted.index, aspect_stats_sorted['mention_rate'], color=colors_aspect)
    plt.xlabel('Mention Rate (%)', fontsize=12)
    plt.title('ABSA: Aspect Mention Rates', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/4_absa_mention_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. TextRank关键词词云
    print("5. TextRank关键词可视化...")
    plt.figure(figsize=(14, 8))
    if len(keywords_df) > 0:
        keywords_top20 = keywords_df.head(20)
        plt.barh(range(len(keywords_top20)), keywords_top20.iloc[:, 1].values)
        plt.yticks(range(len(keywords_top20)), keywords_top20.iloc[:, 0].values)
        plt.xlabel('Score/Frequency', fontsize=12)
        plt.title('Top 20 Keywords (TextRank)', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/5_textrank_keywords.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 品牌提及
    print("6. 品牌提及统计...")
    plt.figure(figsize=(12, 6))
    if len(brand_df) > 0:
        top_brands = brand_df.head(10)
        plt.bar(top_brands['brand'], top_brands['count'], color='steelblue')
        plt.xlabel('Brand', fontsize=12)
        plt.ylabel('Mention Count', fontsize=12)
        plt.title('Top 10 Brand Mentions in Reviews', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/6_brand_mentions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 材质提及
    print("7. 材质提及统计...")
    plt.figure(figsize=(12, 6))
    if len(material_df) > 0:
        top_materials = material_df.head(10)
        plt.bar(top_materials['material'], top_materials['count'], color='coral')
        plt.xlabel('Material', fontsize=12)
        plt.ylabel('Mention Count', fontsize=12)
        plt.title('Top 10 Material Mentions in Reviews', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/7_material_mentions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 评论长度分布
    print("8. 评论长度分布...")
    plt.figure(figsize=(12, 6))
    plt.hist(reviews_df['text_len'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Review Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Review Length Distribution', fontsize=16, fontweight='bold')
    plt.axvline(reviews_df['text_len'].median(), color='red', linestyle='--', 
                label=f'Median: {reviews_df["text_len"].median():.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{Config.VIZ_DIR}/8_review_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 所有图表已保存到: {Config.VIZ_DIR}/")
    print(f"  共生成 8 张可视化图表")

# ==================== 生成报告 ====================
def generate_summary_report(reviews_df, aspect_stats, keywords_df, brand_df, material_df):
    """生成分析摘要报告"""
    print("\n" + "="*80)
    print("【第八步：生成分析报告】")
    print("="*80)
    
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_summary': {
            'total_reviews': int(len(reviews_df)),
            'avg_review_length': float(reviews_df['text_len'].mean()),
            'date_range': {
                'start': str(reviews_df['review_date_dt'].min()),
                'end': str(reviews_df['review_date_dt'].max())
            }
        },
        'bert_sentiment': {
            'positive': int((reviews_df['bert_label'] == 'POSITIVE').sum()),
            'negative': int((reviews_df['bert_label'] == 'NEGATIVE').sum()),
            'positive_rate': float((reviews_df['bert_label'] == 'POSITIVE').sum() / len(reviews_df) * 100)
        },
        'absa_insights': {
            'most_mentioned_aspect': aspect_stats.index[0],
            'most_positive_aspect': aspect_stats['avg_sentiment'].idxmax(),
            'most_negative_aspect': aspect_stats['avg_sentiment'].idxmin(),
            'aspect_stats': aspect_stats.to_dict()
        },
        'top_keywords': keywords_df.head(10).to_dict('records'),
        'top_brands': brand_df.head(5).to_dict('records'),
        'top_materials': material_df.head(5).to_dict('records')
    }
    
    # 保存JSON报告
    with open(f"{Config.OUTPUT_DIR}/analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_report = f"""# Amazon Kitchen Knife Reviews - NLP Analysis Report

**Generated**: {report['analysis_date']}

## Data Summary
- Total Reviews: {report['data_summary']['total_reviews']:,}
- Average Review Length: {report['data_summary']['avg_review_length']:.1f} characters
- Date Range: {report['data_summary']['date_range']['start']} to {report['data_summary']['date_range']['end']}

## BERT Sentiment Analysis
- Positive Reviews: {report['bert_sentiment']['positive']:,} ({report['bert_sentiment']['positive_rate']:.1f}%)
- Negative Reviews: {report['bert_sentiment']['negative']:,}

## ABSA Insights
- Most Mentioned Aspect: **{report['absa_insights']['most_mentioned_aspect']}**
- Most Positive Aspect: **{report['absa_insights']['most_positive_aspect']}**
- Most Negative Aspect: **{report['absa_insights']['most_negative_aspect']}**

## Top Keywords
{chr(10).join([f"{i+1}. {kw['keyword']}" for i, kw in enumerate(report['top_keywords'])])}

## Top Brand Mentions
{chr(10).join([f"{i+1}. {b['brand']}: {b['count']} mentions" for i, b in enumerate(report['top_brands'])])}

## Top Material Mentions
{chr(10).join([f"{i+1}. {m['material']}: {m['count']} mentions" for i, m in enumerate(report['top_materials'])])}

---
*Report generated by Amazon Review NLP Analysis System*
"""
    
    with open(f"{Config.OUTPUT_DIR}/ANALYSIS_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"✓ JSON报告已保存: {Config.OUTPUT_DIR}/analysis_summary.json")
    print(f"✓ Markdown报告已保存: {Config.OUTPUT_DIR}/ANALYSIS_REPORT.md")
    
    return report

# ==================== 主函数 ====================
def main():
    """主执行函数"""
    print("\n" + "="*80)
    print("🔍 亚马逊厨刀评论NLP完整分析系统")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 0. 创建目录
    setup_directories()
    
    # 1. 加载数据
    reviews_df, fact_reviews, products = load_data()
    
    # 2. BERT情感分析
    reviews_df = bert_sentiment_analysis(reviews_df)
    
    # 3. ABSA方面级情感
    aspect_df, aspect_stats = absa_analysis(reviews_df)
    
    # 4. TextRank关键词
    keywords_df = textrank_keywords(reviews_df)
    
    # 5. 简单NER
    brand_df, material_df = simple_ner(reviews_df)
    
    # 6. 主题建模
    lda_topics, nmf_topics = topic_modeling(reviews_df)
    
    # 7. 生成可视化
    create_visualizations(reviews_df, aspect_stats, keywords_df, brand_df, material_df)
    
    # 8. 生成报告
    report = generate_summary_report(reviews_df, aspect_stats, keywords_df, brand_df, material_df)
    
    print("\n" + "="*80)
    print("✅ 所有分析完成!")
    print("="*80)
    print(f"\n结果保存位置:")
    print(f"  📁 主目录: {Config.OUTPUT_DIR}/")
    print(f"  📊 数据文件: {Config.DATA_DIR}/")
    print(f"  📈 可视化图表: {Config.VIZ_DIR}/")
    print(f"\n生成的文件:")
    print(f"  - analysis_summary.json (完整分析结果)")
    print(f"  - ANALYSIS_REPORT.md (分析报告)")
    print(f"  - 8张可视化图表")
    print(f"  - 7个CSV数据文件")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
