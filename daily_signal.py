#!/usr/bin/env python3
"""
每日选股信号推送脚本
每日收盘后自动扫描，生成BUY信号列表，定时发送推送
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_strategy import StockSelector, StockSignal
from notifier import NotificationManager, load_config


def load_watchlist(watchlist_file: str = None) -> Dict[str, Dict]:
    """
    加载股票池配置
    
    Args:
        watchlist_file: 股票池文件路径
        
    Returns:
        Dict: 股票池字典 {代码: {name, category, enabled}}
    """
    if watchlist_file and os.path.exists(watchlist_file):
        try:
            with open(watchlist_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载股票池失败: {e}")
    
    # 默认股票池
    return {
        "600000": {"name": "浦发银行", "category": "银行", "enabled": True},
        "600036": {"name": "招商银行", "category": "银行", "enabled": True},
        "600016": {"name": "民生银行", "category": "银行", "enabled": True},
        "600015": {"name": "华夏银行", "category": "银行", "enabled": True},
        "600030": {"name": "中信证券", "category": "证券", "enabled": True},
        "600012": {"name": "皖通高速", "category": "高速", "enabled": True},
        "600033": {"name": "福建高速", "category": "高速", "enabled": True},
        "600035": {"name": "宁沪高速", "category": "高速", "enabled": True},
        "600009": {"name": "上海机场", "category": "机场", "enabled": True},
        "600085": {"name": "同仁堂", "category": "医药", "enabled": True},
        "600352": {"name": "山东黄金", "category": "黄金", "enabled": True},
    }


def scan_market(
    selector: StockSelector,
    watchlist: Dict[str, Dict],
    config: Optional[Dict] = None
) -> Dict[str, List[StockSignal]]:
    """
    扫描市场，生成信号报告
    
    Args:
        selector: 选股器实例
        watchlist: 股票池
        config: 扫描配置
        
    Returns:
        Dict: {'buy': [], 'sell': [], 'hold': []}
    """
    config = config or {}
    results = {
        'buy': [],
        'sell': [],
        'hold': []
    }
    
    enabled_stocks = [s for s, c in watchlist.items() if c.get('enabled', True)]
    
    print(f"📊 开始扫描 {len(enabled_stocks)} 只股票...")
    
    for i, symbol in enumerate(enabled_stocks):
        try:
            signal = selector.get_signal(symbol)
            if signal:
                # 更新股票名称
                if symbol in watchlist:
                    signal.name = watchlist[symbol].get('name', signal.name)
                
                # 分类
                if signal.signal == "BUY":
                    results['buy'].append(signal)
                elif signal.signal == "SELL":
                    results['sell'].append(signal)
                else:
                    results['hold'].append(signal)
            
            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(enabled_stocks)}")
                
        except Exception as e:
            print(f"  ⚠️ 处理 {symbol} 时出错: {e}")
            continue
    
    # 按MA20角度排序
    results['buy'].sort(key=lambda x: x.ma20_angle, reverse=True)
    results['sell'].sort(key=lambda x: x.ma20_angle, reverse=True)
    results['hold'].sort(key=lambda x: x.ma20_angle, reverse=True)
    
    return results


def format_signal_for_export(signal: StockSignal) -> Dict:
    """将信号转换为可导出的字典"""
    return {
        'symbol': signal.symbol,
        'name': signal.name,
        'price': round(signal.price, 2),
        'change_pct': round(signal.change_pct, 2),
        'ma20': round(signal.ma20, 2),
        'ma20_angle': round(signal.ma20_angle, 2),
        'rsi': round(signal.rsi, 1),
        'rsi_signal': signal.rsi_signal,
        'macd_signal': signal.macd_signal,
        'signal': signal.signal,
        'signal_desc': signal.signal_desc,
        'update_time': signal.update_time
    }


def generate_summary(results: Dict) -> str:
    """
    生成扫描总结
    
    Args:
        results: 扫描结果
        
    Returns:
        str: 总结文本
    """
    buy_count = len(results['buy'])
    sell_count = len(results['sell'])
    hold_count = len(results['hold'])
    
    # 找到最强和最弱
    if results['buy']:
        strongest = results['buy'][0]
        summary_lines = [
            f"扫描完成！共 {buy_count + sell_count + hold_count} 只股票",
            f"",
            f"🟢 买入信号: {buy_count} 只",
            f"  最强信号: {strongest.name}({strongest.symbol}) MA20角{strongest.ma20_angle:.2f}°"
        ]
    else:
        summary_lines = [
            f"扫描完成！共 {buy_count + sell_count + hold_count} 只股票",
            f"🟢 买入信号: {buy_count} 只",
        ]
    
    if results['sell']:
        weakest = results['sell'][0]
        summary_lines.append(f"🔴 卖出信号: {sell_count} 只")
        summary_lines.append(f"  最弱信号: {weakest.name}({weakest.symbol}) MA20角{weakest.ma20_angle:.2f}°")
    
    summary_lines.append(f"🟡 观望信号: {hold_count} 只")
    
    return "\n".join(summary_lines)


def send_notifications(
    notifier: NotificationManager,
    results: Dict,
    summary: str,
    dry_run: bool = False
) -> Dict:
    """
    发送推送通知
    
    Args:
        notifier: 通知管理器
        results: 扫描结果
        summary: 总结文本
        dry_run: 干跑模式（不实际发送）
        
    Returns:
        Dict: 发送结果
    """
    if dry_run:
        print("🌀 干跑模式，跳过实际推送")
        print(f"📧 邮件: {'已配置' if notifier.email_notifier.is_configured() else '未配置'}")
        print(f"📱 飞书: {'已配置' if notifier.feishu_notifier.is_configured() else '未配置'}")
        return {'dry_run': True}
    
    # 转换结果格式
    buy_signals = [format_signal_for_export(s) for s in results['buy']]
    sell_signals = [format_signal_for_export(s) for s in results['sell']]
    
    # 发送每日报告
    send_results = notifier.send_daily_report(
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        summary=summary
    )
    
    return send_results


def save_results(
    results: Dict,
    output_file: str = None
):
    """
    保存扫描结果到文件
    
    Args:
        results: 扫描结果
        output_file: 输出文件路径
    """
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"signals_{timestamp}.json"
    
    # 转换结果
    output = {
        'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'buy_signals': [format_signal_for_export(s) for s in results['buy']],
        'sell_signals': [format_signal_for_export(s) for s in results['sell']],
        'hold_signals': [format_signal_for_export(s) for s in results['hold'][:20]]  # 只保存前20只
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")


def print_report(results: Dict):
    """
    打印扫描报告
    
    Args:
        results: 扫描结果
    """
    print("\n" + "=" * 70)
    print("📊 每日选股信号扫描报告")
    print(f"🕐 扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 买入信号
    print(f"\n🟢 买入信号 ({len(results['buy'])} 只)")
    print("-" * 70)
    if results['buy']:
        print(f"{'代码':<10} {'名称':<10} {'价格':<10} {'涨幅':<10} {'MA20角':<10} {'RSI':<8} {'描述'}")
        print("-" * 70)
        for s in results['buy']:
            print(f"{s.symbol:<10} {s.name:<10} {s.price:<10.2f} {s.change_pct:<+10.2f}% {s.ma20_angle:<10.2f}° {s.rsi:<8.1f} {s.signal_desc}")
    else:
        print("  暂无买入信号")
    
    # 卖出信号
    print(f"\n🔴 卖出信号 ({len(results['sell'])} 只)")
    print("-" * 70)
    if results['sell']:
        print(f"{'代码':<10} {'名称':<10} {'价格':<10} {'涨幅':<10} {'MA20角':<10} {'RSI':<8} {'描述'}")
        print("-" * 70)
        for s in results['sell']:
            print(f"{s.symbol:<10} {s.name:<10} {s.price:<10.2f} {s.change_pct:<+10.2f}% {s.ma20_angle:<10.2f}° {s.rsi:<8.1f} {s.signal_desc}")
    else:
        print("  暂无卖出信号")
    
    # 观望信号
    print(f"\n🟡 观望信号 ({len(results['hold'])} 只)")
    print("-" * 70)
    if results['hold']:
        for s in results['hold'][:10]:
            print(f"  {s.symbol} {s.name}: 价格{s.price:.2f} MA20角{s.ma20_angle:.2f}°")
        if len(results['hold']) > 10:
            print(f"  ... 还有 {len(results['hold']) - 10} 只")
    else:
        print("  暂无观望信号")
    
    print("\n" + "=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='每日选股信号扫描和推送')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='配置文件路径 (default: config.yaml)')
    parser.add_argument('--watchlist', '-w', type=str,
                        help='股票池配置文件路径')
    parser.add_argument('--output', '-o', type=str,
                        help='输出文件路径 (JSON格式)')
    parser.add_argument('--dry-run', action='store_true',
                        help='干跑模式，不实际发送推送')
    parser.add_argument('--no-push', action='store_true',
                        help='不发送推送，只打印报告')
    parser.add_argument('--angle-threshold', type=float,
                        help='自定义买入角度阈值')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 quant_project 每日选股信号扫描")
    print(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载股票池
    watchlist = load_watchlist(args.watchlist)
    print(f"📋 已加载 {len(watchlist)} 只股票")
    
    # 初始化选股器
    selector_config = {}
    if args.angle_threshold:
        selector_config['angle_threshold_buy'] = args.angle_threshold
    selector = StockSelector(selector_config)
    selector.set_watchlist(watchlist)
    
    # 扫描市场
    results = scan_market(selector, watchlist, config.get('scan'))
    
    # 生成总结
    summary = generate_summary(results)
    
    # 打印报告
    print_report(results)
    print(f"\n{summary}")
    
    # 保存结果
    if args.output:
        save_results(results, args.output)
    
    # 发送推送
    if not args.no_push:
        notifier = NotificationManager(config)
        send_results = send_notifications(notifier, results, summary, args.dry_run)
        
        if not args.dry_run:
            if send_results.get('email'):
                print("✅ 邮件推送成功")
            if send_results.get('feishu'):
                print("✅ 飞书推送成功")
            
            if not send_results.get('email') and not send_results.get('feishu'):
                print("⚠️ 未发送任何推送，请检查配置")
    
    print("\n✨ 扫描完成！")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ==================== 使用说明 ====================

"""
使用示例:

1. 基础扫描（打印报告）:
   python daily_signal.py

2. 扫描并发送推送:
   python daily_signal.py --config config.yaml

3. 干跑模式（不发送）:
   python daily_signal.py --dry-run

4. 只打印报告，不发送:
   python daily_signal.py --no-push

5. 自定义角度阈值:
   python daily_signal.py --angle-threshold 5.0

6. 保存结果到文件:
   python daily_signal.py -o signals_20260213.json

定时任务设置 (crontab):
# 每天 16:00 执行（收盘后）
0 16 * * 1-5 cd /path/to/quant_project && python daily_signal.py --config config.yaml

# 或者使用系统级定时
0 16 * * 1-5 /usr/bin/python3 /path/to/quant_project/daily_signal.py -c /path/to/config.yaml >> /var/log/daily_signals.log 2>&1
"""
