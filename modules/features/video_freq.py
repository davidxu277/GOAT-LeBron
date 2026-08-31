
import pandas as pd

class VideoFrequencyBucket:
    """把视频出现次数分桶 —— 冷门视频自己的 embedding 学不动，
    但"它是个冷门视频"这件事本身就有信息量。"""
    def __init__(self, config):
        cfg = config["features"]["视频热度分桶"]
        self.field = cfg["field"]
        self.edges = cfg["edges"]
        self.counts = None
    def fit(self, train_df):
        self.counts = train_df[self.field].value_counts()      # 只看训练集（R2）
    def transform(self, df):
        df = df.copy()
        freq = df[self.field].map(self.counts).fillna(0)
        df["视频热度桶"] = freq.map(lambda v: sum(v > e for e in self.edges)).astype("int64")
        return df
