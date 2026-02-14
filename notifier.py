#!/usr/bin/env python3
"""
推送通知模块
支持邮件推送和飞书Webhook推送
集成推送频率控制，避免频繁打扰
"""

import os
import smtplib
import json
import time
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from typing import Optional, List, Dict, Any
import urllib.request
import urllib.parse
import urllib.error

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationManager:
    """推送通知管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化通知管理器
        
        Args:
            config: 配置字典，包含邮件和飞书配置
        """
        self.config = config or {}
        self.email_notifier = EmailNotifier(self.config.get('email', {}))
        self.feishu_notifier = FeishuNotifier(self.config.get('feishu', {}))
        self.rate_limiter = RateLimiter(
            max_per_minute=self.config.get('rate_limit', {}).get('max_per_minute', 3),
            max_per_hour=self.config.get('rate_limit', {}).get('max_per_hour', 20),
            cooldown_seconds=self.config.get('rate_limit', {}).get('cooldown_seconds', 300)
        )
    
    def send_stock_signal(
        self,
        symbol: str,
        name: str,
        signal: str,
        price: float,
        change_pct: float,
        ma20_angle: float,
        rsi: float,
        macd_signal: str,
        **kwargs
    ) -> Dict[str, bool]:
        """
        发送股票信号通知
        
        Args:
            symbol: 股票代码
            name: 股票名称
            signal: 信号类型 (BUY/SELL/HOLD)
            price: 当前价格
            change_pct: 涨跌幅
            ma20_angle: MA20角度
            rsi: RSI值
            macd_signal: MACD信号
            **kwargs: 其他扩展参数
            
        Returns:
            Dict: 各渠道发送结果
        """
        results = {}
        
        # 检查频率限制
        rate_limit_result = self.rate_limiter.check(symbol, signal)
        if not rate_limit_result['allowed']:
            logger.info(f"⏸️ 频率限制触发，跳过推送: {symbol} {signal}")
            results['rate_limited'] = True
            return results
        
        results['rate_limited'] = False
        
        # 准备消息内容
        content = self._format_signal_content(
            symbol, name, signal, price, change_pct, ma20_angle, rsi, macd_signal, **kwargs
        )
        
        # 邮件推送
        if self.email_notifier.is_configured():
            email_result = self.email_notifier.send_signal(
                symbol=symbol,
                name=name,
                signal=signal,
                price=price,
                change_pct=change_pct,
                ma20_angle=ma20_angle,
                rsi=rsi,
                macd_signal=macd_signal,
                **kwargs
            )
            results['email'] = email_result['success']
            if email_result.get('success'):
                logger.info(f"✅ 邮件推送成功: {symbol} {signal}")
        
        # 飞书推送
        if self.feishu_notifier.is_configured():
            feishu_result = self.feishu_notifier.send_card(
                title=f"选股信号提醒 - {signal}",
                content=content,
                signal=signal,
                symbol=symbol,
                **kwargs
            )
            results['feishu'] = feishu_result['success']
            if feishu_result.get('success'):
                logger.info(f"✅ 飞书推送成功: {symbol} {signal}")
        
        return results
    
    def send_daily_report(
        self,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        summary: str
    ) -> Dict[str, bool]:
        """
        发送每日选股报告
        
        Args:
            buy_signals: 买入信号列表
            sell_signals: 卖出信号列表
            summary: 总结文本
            
        Returns:
            Dict: 各渠道发送结果
        """
        results = {}
        
        # 邮件推送
        if self.email_notifier.is_configured():
            email_result = self.email_notifier.send_daily_report(
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                summary=summary
            )
            results['email'] = email_result['success']
        
        # 飞书推送
        if self.feishu_notifier.is_configured():
            content = self._format_daily_content(buy_signals, sell_signals, summary)
            feishu_result = self.feishu_notifier.send_card(
                title="📊 每日选股信号报告",
                content=content,
                signal="REPORT"
            )
            results['feishu'] = feishu_result['success']
        
        return results
    
    def _format_signal_content(
        self,
        symbol: str,
        name: str,
        signal: str,
        price: float,
        change_pct: float,
        ma20_angle: float,
        rsi: float,
        macd_signal: str,
        **kwargs
    ) -> str:
        """格式化信号消息内容"""
        emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        
        content = f"""**{emoji} {signal}信号提醒**

📈 {name} ({symbol})
💰 当前价格: {price:.2f} ({change_pct:+.2f}%)

📊 技术指标:
• MA20角度: **{ma20_angle:.2f}°**
• RSI: **{rsi:.1f}** ({'超买' if rsi >= 70 else '超卖' if rsi <= 30 else '中性'})
• MACD: **{macd_signal}**

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return content
    
    def _format_daily_content(
        self,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        summary: str
    ) -> str:
        """格式化每日报告内容"""
        content = f"""**📊 每日选股信号报告**

📅 {datetime.now().strftime('%Y-%m-%d')}

🟢 **买入信号**: {len(buy_signals)} 只
"""
        
        if buy_signals:
            content += "\n| 代码 | 名称 | 价格 | 涨幅 | MA20角 | RSI |\n"
            content += "|------|------|------|------|--------|-----|\n"
            for s in buy_signals[:10]:  # 最多显示10只
                content += f"| {s.get('symbol', '-')} | {s.get('name', '-')} | {s.get('price', 0):.2f} | {s.get('change_pct', 0):+.2f}% | {s.get('ma20_angle', 0):.2f}° | {s.get('rsi', 0):.1f} |\n"
        
        if len(buy_signals) > 10:
            content += f"\n... 还有 {len(buy_signals) - 10} 只买入信号\n"
        
        content += f"""

🔴 **卖出信号**: {len(sell_signals)} 只

📝 **总结**: {summary}

---
*由 quant_project 自动生成*
"""
        return content


class EmailNotifier:
    """邮件推送通知器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化邮件推送器
        
        Args:
            config: 配置字典，包含 SMTP 配置
                - smtp_host: SMTP 服务器地址
                - smtp_port: SMTP 端口
                - username: 邮箱用户名
                - password: 邮箱密码/授权码
                - recipients: 收件人列表
                - sender_name: 发件人显示名称
        """
        self.config = config or {}
        self.smtp_host = self.config.get('smtp_host', '')
        self.smtp_port = self.config.get('smtp_port', 465)
        self.username = self.config.get('username', '')
        self.password = self.config.get('password', '')
        self.recipients = self.config.get('recipients', [])
        self.sender_name = self.config.get('sender_name', 'Quant Signals')
    
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.smtp_host and self.username and self.password and self.recipients)
    
    def send_signal(
        self,
        symbol: str,
        name: str,
        signal: str,
        price: float,
        change_pct: float,
        ma20_angle: float,
        rsi: float,
        macd_signal: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送股票信号邮件
        
        Args:
            symbol: 股票代码
            name: 股票名称
            signal: 信号类型
            price: 当前价格
            change_pct: 涨跌幅
            ma20_angle: MA20角度
            rsi: RSI值
            macd_signal: MACD信号
            
        Returns:
            Dict: 发送结果
        """
        if not self.is_configured():
            return {'success': False, 'error': '邮件配置不完整'}
        
        try:
            # 构建邮件内容
            subject = f"【{signal}信号】{name} ({symbol}) - MA20角{ma20_angle:.2f}°"
            html_content = self._build_signal_html(
                symbol, name, signal, price, change_pct, ma20_angle, rsi, macd_signal, **kwargs
            )
            
            # 发送邮件
            result = self._send_email(
                subject=subject,
                html_content=html_content,
                recipients=self.recipients
            )
            
            return {'success': result, 'error': None if result else '发送失败'}
            
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_daily_report(
        self,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        summary: str
    ) -> Dict[str, Any]:
        """
        发送每日选股报告邮件
        
        Args:
            buy_signals: 买入信号列表
            sell_signals: 卖出信号列表
            summary: 总结
            
        Returns:
            Dict: 发送结果
        """
        if not self.is_configured():
            return {'success': False, 'error': '邮件配置不完整'}
        
        try:
            subject = f"📊 每日选股报告 - {datetime.now().strftime('%Y-%m-%d')}"
            html_content = self._build_daily_report_html(buy_signals, sell_signals, summary)
            
            result = self._send_email(
                subject=subject,
                html_content=html_content,
                recipients=self.recipients
            )
            
            return {'success': result, 'error': None if result else '发送失败'}
            
        except Exception as e:
            logger.error(f"❌ 每日报告邮件发送失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_signal_html(
        self,
        symbol: str,
        name: str,
        signal: str,
        price: float,
        change_pct: float,
        ma20_angle: float,
        rsi: float,
        macd_signal: str,
        **kwargs
    ) -> str:
        """构建股票信号HTML邮件内容"""
        emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        bg_color = "#e8f5e9" if signal == "BUY" else "#ffebee" if signal == "SELL" else "#fff8e1"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: {bg_color}; padding: 20px; border-radius: 0 0 10px 10px; }}
        .signal-badge {{ display: inline-block; padding: 10px 20px; border-radius: 20px; font-size: 18px; font-weight: bold; }}
        .buy {{ background: #4caf50; color: white; }}
        .sell {{ background: #f44336; color: white; }}
        .hold {{ background: #ff9800; color: white; }}
        .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; }}
        .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{emoji} {signal} Signal Alert</h1>
        </div>
        <div class="content">
            <h2>{name} ({symbol})</h2>
            <span class="signal-badge {signal.lower()}">{signal}</span>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{price:.2f}</div>
                    <div class="metric-label">当前价格 ({change_pct:+.2f}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{ma20_angle:.2f}°</div>
                    <div class="metric-label">MA20 角度</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{rsi:.1f}</div>
                    <div class="metric-label">RSI (14)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{macd_signal}</div>
                    <div class="metric-label">MACD 信号</div>
                </div>
            </div>
            
            <p style="margin-top: 20px; color: #666;">
                生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
        <div class="footer">
            Powered by quant_project
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _build_daily_report_html(
        self,
        buy_signals: List[Dict],
        sell_signals: List[Dict],
        summary: str
    ) -> str:
        """构建每日报告HTML邮件内容"""
        buy_rows = ""
        for s in buy_signals[:15]:
            buy_rows += f"""
            <tr>
                <td>{s.get('symbol', '-')}</td>
                <td>{s.get('name', '-')}</td>
                <td>{s.get('price', 0):.2f}</td>
                <td>{s.get('change_pct', 0):+.2f}%</td>
                <td>{s.get('ma20_angle', 0):.2f}°</td>
                <td>{s.get('rsi', 0):.1f}</td>
            </tr>
"""
        
        sell_rows = ""
        for s in sell_signals[:15]:
            sell_rows += f"""
            <tr>
                <td>{s.get('symbol', '-')}</td>
                <td>{s.get('name', '-')}</td>
                <td>{s.get('price', 0):.2f}</td>
                <td>{s.get('change_pct', 0):+.2f}%</td>
                <td>{s.get('ma20_angle', 0):.2f}°</td>
                <td>{s.get('rsi', 0):.1f}</td>
            </tr>
"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }}
        .content {{ background: #f9f9f9; padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: white; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        .buy-section {{ margin-bottom: 30px; }}
        .sell-section {{ margin-bottom: 30px; }}
        .buy-title {{ color: #4caf50; font-size: 18px; margin-bottom: 10px; }}
        .sell-title {{ color: #f44336; font-size: 18px; margin-bottom: 10px; }}
        .summary {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Stock Signal Report</h1>
            <p>{datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        <div class="content">
            <div class="summary">
                <strong>📝 Summary:</strong><br>
                {summary}
            </div>
            
            <div class="buy-section">
                <div class="buy-title">🟢 Buy Signals ({len(buy_signals)} stocks)</div>
                <table>
                    <thead>
                        <tr>
                            <th>Code</th>
                            <th>Name</th>
                            <th>Price</th>
                            <th>Change</th>
                            <th>MA20 Angle</th>
                            <th>RSI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {buy_rows if buy_rows else '<tr><td colspan="6">No buy signals today</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            <div class="sell-section">
                <div class="sell-title">🔴 Sell Signals ({len(sell_signals)} stocks)</div>
                <table>
                    <thead>
                        <tr>
                            <th>Code</th>
                            <th>Name</th>
                            <th>Price</th>
                            <th>Change</th>
                            <th>MA20 Angle</th>
                            <th>RSI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sell_rows if sell_rows else '<tr><td colspan="6">No sell signals today</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        <div class="footer">
            Powered by quant_project
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _send_email(
        self,
        subject: str,
        html_content: str,
        recipients: List[str]
    ) -> bool:
        """
        发送邮件（内部方法）
        
        Args:
            subject: 邮件主题
            html_content: HTML内容
            recipients: 收件人列表
            
        Returns:
            bool: 是否发送成功
        """
        try:
            # 构建邮件
            msg = MIMEText(html_content, 'html', 'utf-8')
            msg['Subject'] = Header(subject, 'utf-8')
            msg['From'] = formataddr([self.sender_name, self.username])
            msg['To'] = ','.join(recipients)
            
            # 连接SMTP服务器并发送
            if self.smtp_port == 465:
                # SSL
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.username, recipients, msg.as_string())
            else:
                # TLS
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(self.username, recipients, msg.as_string())
            
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP认证失败: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP发送失败: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 邮件发送异常: {e}")
            return False


class FeishuNotifier:
    """飞书Webhook推送通知器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化飞书推送器
        
        Args:
            config: 配置字典
                - webhook_url: 飞书Webhook地址
                - mention_users: @提醒的用户列表 (open_id)
        """
        self.config = config or {}
        self.webhook_url = self.config.get('webhook_url', '')
        self.mention_users = self.config.get('mention_users', [])
    
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.webhook_url)
    
    def send_card(
        self,
        title: str,
        content: str,
        signal: str = "INFO",
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送飞书卡片消息
        
        Args:
            title: 标题
            content: 内容（支持Markdown）
            signal: 信号类型（用于确定颜色）
            **kwargs: 其他扩展参数
            
        Returns:
            Dict: 发送结果
        """
        if not self.is_configured():
            return {'success': False, 'error': '飞书Webhook未配置'}
        
        try:
            # 确定颜色
            color_map = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'yellow',
                'REPORT': 'blue'
            }
            color = color_map.get(signal, 'gray')
            
            # 构建消息
            payload = {
                "msg_type": "card",
                "card": {
                    "config": {
                        "wide_screen_mode": True
                    },
                    "elements": [
                        {
                            "tag": "div",
                            "fields": [
                                {
                                    "is_short": True,
                                    "text": {
                                        "type": "markdown",
                                        "content": f"**{title}**"
                                    }
                                },
                                {
                                    "is_short": True,
                                    "text": {
                                        "type": "markdown",
                                        "content": f"**类型**: {signal}"
                                    }
                                }
                            ]
                        },
                        {
                            "tag": "div",
                            "text": {
                                "type": "markdown",
                                "content": content
                            }
                        },
                        {
                            "tag": "action",
                            "actions": [
                                {
                                    "tag": "button",
                                    "text": {
                                        "tag": "plain_text",
                                        "content": "📊 查看详情"
                                    },
                                    "type": "primary",
                                    "url": "https://github.com/zhangjc138/quant_project"
                                }
                            ]
                        }
                    ]
                }
            }
            
            # 发送请求
            result = self._send_request(payload)
            
            if result.get('code') == 0:
                return {'success': True}
            else:
                logger.error(f"❌ 飞书推送失败: {result}")
                return {'success': False, 'error': result.get('msg', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"❌ 飞书推送异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_text(self, text: str) -> Dict[str, Any]:
        """
        发送飞书文本消息
        
        Args:
            text: 文本内容
            
        Returns:
            Dict: 发送结果
        """
        if not self.is_configured():
            return {'success': False, 'error': '飞书Webhook未配置'}
        
        try:
            # 构建消息
            payload = {
                "msg_type": "text",
                "content": {
                    "text": text
                }
            }
            
            # @提醒用户
            if self.mention_users:
                at_text = ""
                for user_id in self.mention_users:
                    at_text += f"<at user_id=\"{user_id}\"></at>"
                payload['content']['text'] = at_text + text
            
            result = self._send_request(payload)
            
            return {'success': result.get('code') == 0}
            
        except Exception as e:
            logger.error(f"❌ 飞书文本推送异常: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_request(self, payload: Dict) -> Dict:
        """
        发送HTTP请求（内部方法）
        
        Args:
            payload: 消息载荷
            
        Returns:
            Dict: 响应结果
        """
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))


class RateLimiter:
    """
    推送频率限制器
    
    防止同一天内对同一股票发送过多推送
    """
    
    def __init__(
        self,
        max_per_minute: int = 3,
        max_per_hour: int = 20,
        cooldown_seconds: int = 300
    ):
        """
        初始化频率限制器
        
        Args:
            max_per_minute: 每分钟最大推送数
            max_per_hour: 每小时最大推送数
            cooldown_seconds: 同股票同信号冷却时间（秒）
        """
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.cooldown_seconds = cooldown_seconds
        
        # 记录
        self.minute_history = []  # [(timestamp, symbol, signal)]
        self.hour_history = []
        self.cooldown_cache = {}  # {(symbol, signal): timestamp}
    
    def check(self, symbol: str, signal: str) -> Dict[str, Any]:
        """
        检查是否允许推送
        
        Args:
            symbol: 股票代码
            signal: 信号类型
            
        Returns:
            Dict: {'allowed': bool, 'reason': str}
        """
        now = time.time()
        key = (symbol, signal)
        
        # 清理过期记录
        self.minute_history = [
            (ts, s, sg) for ts, s, sg in self.minute_history
            if now - ts < 60
        ]
        self.hour_history = [
            (ts, s, sg) for ts, s, sg in self.hour_history
            if now - ts < 3600
        ]
        
        # 检查冷却期
        if key in self.cooldown_cache:
            last_time = self.cooldown_cache[key]
            if now - last_time < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - (now - last_time))
                return {
                    'allowed': False,
                    'reason': f'Cooldown: {remaining}s remaining'
                }
        
        # 检查每分钟限制
        minute_count = sum(1 for ts, s, sg in self.minute_history if s == symbol)
        if minute_count >= self.max_per_minute:
            return {
                'allowed': False,
                'reason': 'Minute rate limit exceeded'
            }
        
        # 检查每小时限制
        hour_count = sum(1 for ts, s, sg in self.hour_history if s == symbol)
        if hour_count >= self.max_per_hour:
            return {
                'allowed': False,
                'reason': 'Hourly rate limit exceeded'
            }
        
        # 记录并允许
        self.minute_history.append((now, symbol, signal))
        self.hour_history.append((now, symbol, signal))
        self.cooldown_cache[key] = now
        
        return {'allowed': True, 'reason': None}
    
    def reset(self):
        """重置所有记录"""
        self.minute_history = []
        self.hour_history = []
        self.cooldown_cache = {}


def load_config(config_path: str = None) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（支持 .yaml, .json）
        
    Returns:
        Dict: 配置字典
    """
    import yaml
    
    if config_path is None:
        # 查找默认配置文件
        possible_paths = [
            'config.yaml',
            'config.json',
            'config.yml',
            'notifier_config.yaml',
            'notifier_config.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.warning(f"⚠️ 配置文件未找到: {config_path}")
        return {}
    
    # 根据扩展名解析
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        logger.error(f"❌ 不支持的文件格式: {config_path}")
        return {}
    
    return config


# ==================== 便捷函数 ====================

def create_notifier(config_path: str = None) -> NotificationManager:
    """
    创建通知管理器的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        NotificationManager: 通知管理器实例
    """
    config = load_config(config_path)
    return NotificationManager(config)


def send_signal_notification(
    symbol: str,
    name: str,
    signal: str,
    price: float,
    change_pct: float,
    ma20_angle: float,
    rsi: float,
    macd_signal: str,
    config_path: str = None
) -> Dict[str, bool]:
    """
    发送信号通知的便捷函数
    
    Args:
        symbol: 股票代码
        name: 股票名称
        signal: 信号类型
        price: 当前价格
        change_pct: 涨跌幅
        ma20_angle: MA20角度
        rsi: RSI值
        macd_signal: MACD信号
        config_path: 配置文件路径
        
    Returns:
        Dict: 发送结果
    """
    notifier = create_notifier(config_path)
    return notifier.send_stock_signal(
        symbol=symbol,
        name=name,
        signal=signal,
        price=price,
        change_pct=change_pct,
        ma20_angle=ma20_angle,
        rsi=rsi,
        macd_signal=macd_signal
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试邮件配置
    test_email_config = {
        'smtp_host': 'smtp.qq.com',
        'smtp_port': 465,
        'username': '6315489@qq.com',
        'password': 'your_app_password',  # 需要替换为实际授权码
        'recipients': ['user@example.com'],
        'sender_name': 'Quant Signals'
    }
    
    # 测试飞书配置
    test_feishu_config = {
        'webhook_url': 'https://open.feishu.cn/open-apis/bot/v2/xxx'
    }
    
    print("=" * 50)
    print("notifier.py 模块加载成功")
    print("=" * 50)
    print("\n可用类:")
    print("  - NotificationManager: 统一通知管理")
    print("  - EmailNotifier: 邮件推送")
    print("  - FeishuNotifier: 飞书Webhook推送")
    print("  - RateLimiter: 频率限制")
    print("\n使用示例:")
    print("""
    from notifier import NotificationManager
    
    # 从配置文件加载
    notifier = NotificationManager(config_path='config.yaml')
    
    # 发送信号
    notifier.send_stock_signal(
        symbol='600000',
        name='浦发银行',
        signal='BUY',
        price=12.34,
        change_pct=2.5,
        ma20_angle=5.2,
        rsi=45,
        macd_signal='GOLD_CROSS'
    )
    """)
