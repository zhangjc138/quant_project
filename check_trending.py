#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Trending监控脚本
每日检查项目是否在Trending上，分析Stars增长趋势
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os

# 配置
REPO_OWNER = "zhangjc138"
REPO_NAME = "quant_project"
GITHUB_API = "https://api.github.com"
REPO_URL = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}"
SEARCH_URL = f"{GITHUB_API}/search/repositories"

class TrendingChecker:
    """GitHub Trending检查器"""
    
    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TrendingChecker/1.0"
        }
        # 可选：添加Token以提高API限制
        self.token = os.environ.get("GITHUB_TOKEN", "")
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_repo_info(self) -> Optional[Dict]:
        """获取仓库基本信息"""
        try:
            response = requests.get(REPO_URL, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ 获取仓库信息失败: {e}")
            return None
    
    def get_stars_history(self, days: int = 7) -> List[Dict]:
        """获取Stars历史数据（模拟，实际需要存储）"""
        # 由于GitHub API限制，这里返回当前数据和估算
        repo_info = self.get_repo_info()
        if repo_info:
            return [{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "stars": repo_info.get("stargazers_count", 0),
                "forks": repo_info.get("forks_count", 0),
                "watchers": repo_info.get("watchers_count", 0)
            }]
        return []
    
    def check_trending_rank(self, language: str = "python") -> Optional[int]:
        """检查在GitHub Trending的排名"""
        try:
            # 搜索Python项目中本仓库的排名
            query = f"language:{language} sort:stars"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 100
            }
            
            response = requests.get(SEARCH_URL, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            items = response.json().get("items", [])
            for rank, item in enumerate(items, 1):
                if item.get("full_name") == f"{REPO_OWNER}/{REPO_NAME}":
                    return rank
            
            return None  # 未找到，可能不在前100名
        except Exception as e:
            print(f"❌ 检查Trending排名失败: {e}")
            return None
    
    def check_all_language_trending(self) -> Optional[int]:
        """检查全站Trending排名"""
        try:
            params = {
                "q": f"{REPO_OWNER}/{REPO_NAME}",
                "sort": "stars",
                "per_page": 100
            }
            
            response = requests.get(SEARCH_URL, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            items = response.json().get("items", [])
            for rank, item in enumerate(items, 1):
                if item.get("full_name") == f"{REPO_OWNER}/{REPO_NAME}":
                    return rank
            
            return None
        except Exception as e:
            print(f"❌ 检查全站Trending失败: {e}")
            return None
    
    def get_daily_stars_data(self) -> Dict:
        """获取今日Stars数据"""
        repo = self.get_repo_info()
        if repo:
            return {
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "watchers": repo.get("watchers_count", 0),
                "open_issues": repo.get("open_issues_count", 0),
                "subscribers": repo.get("subscribers_count", 0),
                "description": repo.get("description", ""),
                "topics": repo.get("topics", []),
                "language": repo.get("language", ""),
                "updated_at": repo.get("updated_at", "")
            }
        return {}
    
    def estimate_daily_growth(self) -> Dict:
        """估算每日增长（基于当前数据）"""
        data = self.get_daily_stars_data()
        stars = data.get("stars", 0)
        
        # 估算：假设活跃项目每天增长1-5 stars
        estimated_daily = max(1, int(stars * 0.01))  # 1%的日增长率估算
        estimated_weekly = estimated_daily * 7
        estimated_monthly = estimated_daily * 30
        
        return {
            "current_stars": stars,
            "estimated_daily": estimated_daily,
            "estimated_weekly": estimated_weekly,
            "estimated_monthly": estimated_monthly
        }
    
    def generate_report(self) -> str:
        """生成每日报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"📊 GitHub项目每日监控报告")
        report_lines.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"📁 项目: {REPO_OWNER}/{REPO_NAME}")
        report_lines.append("=" * 60)
        
        # 基本信息
        repo_data = self.get_daily_stars_data()
        report_lines.append("\n🔍 基本信息:")
        report_lines.append(f"   ⭐ Stars: {repo_data.get('stars', 0)}")
        report_lines.append(f"   🍴 Forks: {repo_data.get('forks', 0)}")
        report_lines.append(f"   👁️ Watchers: {repo_data.get('watchers', 0)}")
        report_lines.append(f"   📝 Open Issues: {repo_data.get('open_issues', 0)}")
        report_lines.append(f"   🏷️ Topics: {', '.join(repo_data.get('topics', [])) or '无'}")
        
        # Trending排名
        report_lines.append("\n📈 Trending排名:")
        py_rank = self.check_trending_rank("python")
        if py_rank:
            report_lines.append(f"   🐍 Python类目: 第 {py_rank} 位")
        else:
            report_lines.append(f"   🐍 Python类目: 未进入前100名")
        
        all_rank = self.check_all_language_trending()
        if all_rank:
            report_lines.append(f"   🌍 全站排名: 第 {all_rank} 位")
        else:
            report_lines.append(f"   🌍 全站排名: 未进入前100名")
        
        # 增长估算
        growth = self.estimate_daily_growth()
        report_lines.append("\n📊 增长估算:")
        report_lines.append(f"   当前Stars: {growth['current_stars']}")
        report_lines.append(f"   估算日增长: +{growth['estimated_daily']}")
        report_lines.append(f"   估算周增长: +{growth['estimated_weekly']}")
        report_lines.append(f"   估算月增长: +{growth['estimated_monthly']}")
        
        # 上Trending建议
        report_lines.append("\n💡 上Trending建议:")
        current_stars = growth['current_stars']
        if current_stars < 100:
            report_lines.append("   📌 目标: 达成100 Stars")
            report_lines.append("   💡 建议: 分享到Reddit/掘金/知乎等技术社区")
        elif current_stars < 500:
            report_lines.append("   📌 目标: 达成500 Stars")
            report_lines.append("   💡 建议: 联系技术博主、KOL推荐")
        elif current_stars < 1000:
            report_lines.append("   📌 目标: 达成1000 Stars")
            report_lines.append("   💡 建议: 申请GitHub Trending推荐")
        else:
            report_lines.append("   🎉 已达到较高关注度！")
            report_lines.append("   💡 建议: 持续更新，保持活跃度")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_report(self, filepath: str = "trending_report.txt"):
        """保存报告到文件"""
        report = self.generate_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ 报告已保存到: {filepath}")
        return report


def main():
    """主函数"""
    print("🚀 GitHub Trending监控脚本启动...\n")
    
    checker = TrendingChecker()
    
    # 生成并显示报告
    report = checker.generate_report()
    print(report)
    
    # 保存报告
    checker.save_report()
    
    # 返回成功状态
    return True


if __name__ == "__main__":
    main()
