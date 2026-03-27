"""
伪标签生成模块
根据规则自动生成数据质量标签
"""
from typing import List, Dict, Any, Optional
from loguru import logger


class PseudoLabeler:
    """伪标签生成器"""
    
    # 标签映射
    LABEL_MAP = {
        'high': 2,      # 优质
        'medium': 1,    # 普通
        'low': 0        # 低质
    }
    
    def __init__(
        self,
        like_threshold_high: int = 1000,
        like_threshold_low: int = 100,
        collect_threshold_high: int = 500,
        collect_threshold_low: int = 50,
        comment_threshold_high: int = 100,
        comment_threshold_low: int = 20,
        content_length_high: int = 200,
        content_length_low: int = 50,
        engagement_rate_high: float = 0.05,
        engagement_rate_low: float = 0.01
    ):
        """
        初始化标签生成器
        
        Args:
            like_threshold_high: 高点赞阈值
            like_threshold_low: 低点赞阈值
            collect_threshold_high: 高收藏阈值
            collect_threshold_low: 低收藏阈值
            comment_threshold_high: 高评论阈值
            comment_threshold_low: 低评论阈值
            content_length_high: 长内容阈值
            content_length_low: 短内容阈值
            engagement_rate_high: 高互动率阈值
            engagement_rate_low: 低互动率阈值
        """
        self.like_threshold_high = like_threshold_high
        self.like_threshold_low = like_threshold_low
        self.collect_threshold_high = collect_threshold_high
        self.collect_threshold_low = collect_threshold_low
        self.comment_threshold_high = comment_threshold_high
        self.comment_threshold_low = comment_threshold_low
        self.content_length_high = content_length_high
        self.content_length_low = content_length_low
        self.engagement_rate_high = engagement_rate_high
        self.engagement_rate_low = engagement_rate_low
        
        logger.info("伪标签生成器初始化完成")
    
    def generate_label(self, item: Dict[str, Any]) -> str:
        """
        为单个样本生成标签
        
        Args:
            item: 数据样本
            
        Returns:
            标签 ('high', 'medium', 'low')
        """
        score = 0
        
        # 获取数据
        likes = item.get('likes', 0) or 0
        collects = item.get('collects', 0) or 0
        comments = item.get('comments', 0) or item.get('reply_count', 0) or 0
        views = item.get('views', 1) or 1
        content = item.get('content', '') or item.get('dialogue', [''])[0] or ''
        
        # 点赞数评分 (0-3 分)
        if likes > self.like_threshold_high:
            score += 3
        elif likes > self.like_threshold_low:
            score += 2
        elif likes > 10:
            score += 1
        
        # 收藏数评分 (0-2 分)
        if collects > self.collect_threshold_high:
            score += 2
        elif collects > self.collect_threshold_low:
            score += 1
        
        # 评论数评分 (0-2 分)
        if comments > self.comment_threshold_high:
            score += 2
        elif comments > self.comment_threshold_low:
            score += 1
        
        # 内容长度评分 (0-2 分)
        content_len = len(content)
        if content_len > self.content_length_high:
            score += 2
        elif content_len > self.content_length_low:
            score += 1
        
        # 互动率评分 (0-2 分)
        engagement_rate = (likes + collects) / max(views, 1)
        if engagement_rate > self.engagement_rate_high:
            score += 2
        elif engagement_rate > self.engagement_rate_low:
            score += 1
        
        # 标签判定
        if score >= 8:
            label = 'high'
        elif score >= 5:
            label = 'medium'
        else:
            label = 'low'
        
        return label
    
    def generate_labels(
        self,
        data: List[Dict[str, Any]],
        add_label_id: bool = True
    ) -> List[Dict[str, Any]]:
        """
        为批量数据生成标签
        
        Args:
            data: 数据列表
            add_label_id: 是否添加数字标签 ID
            
        Returns:
            带标签的数据列表
        """
        labeled_data = []
        label_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for item in data:
            label = self.generate_label(item)
            item['label'] = label
            label_counts[label] += 1
            
            if add_label_id:
                item['label_id'] = self.LABEL_MAP[label]
            
            labeled_data.append(item)
        
        # 打印统计
        total = len(data)
        logger.info(f"标签生成完成，共 {total} 条数据")
        logger.info(f"标签分布：high={label_counts['high']} ({label_counts['high']/total*100:.1f}%), "
                   f"medium={label_counts['medium']} ({label_counts['medium']/total*100:.1f}%), "
                   f"low={label_counts['low']} ({label_counts['low']/total*100:.1f}%)")
        
        return labeled_data
    
    def analyze_distribution(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析标签分布
        
        Args:
            data: 已标注数据
            
        Returns:
            分布统计信息
        """
        label_counts = {'high': 0, 'medium': 0, 'low': 0}
        label_likes = {'high': [], 'medium': [], 'low': []}
        
        for item in data:
            label = item.get('label', 'low')
            label_counts[label] += 1
            label_likes[label].append(item.get('likes', 0))
        
        total = len(data)
        
        return {
            'total': total,
            'distribution': {
                label: {
                    'count': count,
                    'percentage': count / total * 100 if total > 0 else 0,
                    'avg_likes': sum(label_likes[label]) / len(label_likes[label]) if label_likes[label] else 0
                }
                for label, count in label_counts.items()
            }
        }
