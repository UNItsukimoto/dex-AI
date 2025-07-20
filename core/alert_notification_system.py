#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
アラート・通知システム
リアルタイム通知とアラート機能の実装
"""

import json
import requests
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import os

# メール機能（オプション）
try:
    import smtplib
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logging.warning("メール機能が利用できません")

# 音声機能（オプション）
try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("音声機能が利用できません")

# デスクトップ通知用
try:
    import plyer
    DESKTOP_NOTIFICATIONS_AVAILABLE = True
except ImportError:
    DESKTOP_NOTIFICATIONS_AVAILABLE = False
    logging.warning("デスクトップ通知が利用できません。pip install plyer で利用可能になります。")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertType(Enum):
    """アラートタイプ"""
    TRADING_SIGNAL = "trading_signal"      # 取引シグナル
    RISK_WARNING = "risk_warning"          # リスク警告
    PRICE_ALERT = "price_alert"            # 価格アラート
    SYSTEM_ERROR = "system_error"          # システムエラー
    PERFORMANCE = "performance"            # パフォーマンス
    MARKET_NEWS = "market_news"            # 市場ニュース

class Priority(Enum):
    """優先度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """アラート情報"""
    id: str
    alert_type: AlertType
    priority: Priority
    title: str
    message: str
    symbol: Optional[str] = None
    timestamp: datetime = None
    data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class NotificationChannel:
    """通知チャンネル基底クラス"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.last_sent = {}
        self.rate_limit = 60  # 秒
    
    def can_send(self, alert_id: str) -> bool:
        """レート制限チェック"""
        if alert_id in self.last_sent:
            elapsed = time.time() - self.last_sent[alert_id]
            return elapsed >= self.rate_limit
        return True
    
    def send(self, alert: Alert) -> bool:
        """通知送信（サブクラスで実装）"""
        if not self.enabled:
            return False
        
        if not self.can_send(alert.id):
            logger.debug(f"{self.name}: レート制限によりスキップ {alert.id}")
            return False
        
        try:
            success = self._send_alert(alert)
            if success:
                self.last_sent[alert.id] = time.time()
            return success
        except Exception as e:
            logger.error(f"{self.name} 通知エラー: {e}")
            return False
    
    def _send_alert(self, alert: Alert) -> bool:
        """実際の送信処理（サブクラスで実装）"""
        raise NotImplementedError

class DesktopNotificationChannel(NotificationChannel):
    """デスクトップ通知"""
    
    def __init__(self, enabled: bool = True):
        super().__init__("Desktop", enabled)
        self.available = DESKTOP_NOTIFICATIONS_AVAILABLE
    
    def _send_alert(self, alert: Alert) -> bool:
        if not self.available:
            return False
        
        try:
            # 優先度に応じたアイコン
            icon_map = {
                Priority.LOW: None,
                Priority.MEDIUM: None,
                Priority.HIGH: None,
                Priority.CRITICAL: None
            }
            
            plyer.notification.notify(
                title=f"[{alert.alert_type.value.upper()}] {alert.title}",
                message=alert.message,
                timeout=10,
                app_icon=icon_map.get(alert.priority)
            )
            
            logger.info(f"デスクトップ通知送信: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"デスクトップ通知エラー: {e}")
            return False

class EmailNotificationChannel(NotificationChannel):
    """メール通知"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 to_emails: List[str], enabled: bool = False):
        super().__init__("Email", enabled and EMAIL_AVAILABLE)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_emails = to_emails
    
    def _send_alert(self, alert: Alert) -> bool:
        if not EMAIL_AVAILABLE:
            return False
            
        try:
            # メール作成
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[AI Trading Alert] {alert.title}"
            
            # メール本文
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # SMTP送信
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"メール送信完了: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"メール送信エラー: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """メール本文HTML作成"""
        priority_colors = {
            Priority.LOW: "#28a745",
            Priority.MEDIUM: "#ffc107", 
            Priority.HIGH: "#fd7e14",
            Priority.CRITICAL: "#dc3545"
        }
        
        color = priority_colors.get(alert.priority, "#6c757d")
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0 0 0;">優先度: {alert.priority.value.upper()}</p>
                </div>
                
                <div style="padding: 20px; border: 1px solid #ddd; border-top: none;">
                    <p><strong>アラートタイプ:</strong> {alert.alert_type.value}</p>
                    <p><strong>時刻:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    {f"<p><strong>銘柄:</strong> {alert.symbol}</p>" if alert.symbol else ""}
                    
                    <div style="margin-top: 15px;">
                        <h3>詳細:</h3>
                        <p>{alert.message}</p>
                    </div>
                    
                    {self._format_alert_data(alert.data) if alert.data else ""}
                </div>
                
                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 0 0 5px 5px; text-align: center;">
                    <small>AI Trading System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_alert_data(self, data: Dict) -> str:
        """アラートデータのフォーマット"""
        if not data:
            return ""
        
        html = "<div style='margin-top: 15px;'><h4>追加データ:</h4><ul>"
        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul></div>"
        
        return html

class SlackNotificationChannel(NotificationChannel):
    """Slack通知"""
    
    def __init__(self, webhook_url: str, enabled: bool = False):
        super().__init__("Slack", enabled)
        self.webhook_url = webhook_url
    
    def _send_alert(self, alert: Alert) -> bool:
        try:
            # Slack色設定
            color_map = {
                Priority.LOW: "good",
                Priority.MEDIUM: "warning",
                Priority.HIGH: "danger", 
                Priority.CRITICAL: "danger"
            }
            
            # Slackメッセージ作成
            payload = {
                "username": "AI Trading Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [{
                    "color": color_map.get(alert.priority, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "アラートタイプ",
                            "value": alert.alert_type.value,
                            "short": True
                        },
                        {
                            "title": "優先度",
                            "value": alert.priority.value.upper(),
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            if alert.symbol:
                payload["attachments"][0]["fields"].append({
                    "title": "銘柄",
                    "value": alert.symbol,
                    "short": True
                })
            
            if alert.data:
                payload["attachments"][0]["fields"].extend([
                    {
                        "title": key,
                        "value": f"{value:.3f}" if isinstance(value, float) else str(value),
                        "short": True
                    }
                    for key, value in alert.data.items()
                ])
            
            # Webhook送信
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack通知送信完了: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Slack通知エラー: {e}")
            return False

class AudioNotificationChannel(NotificationChannel):
    """音声通知"""
    
    def __init__(self, enabled: bool = True):
        super().__init__("Audio", enabled and AUDIO_AVAILABLE)
        self.audio_available = AUDIO_AVAILABLE
        
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.init()
                self.audio_available = True
            except Exception as e:
                logger.warning(f"音声通知が利用できません: {e}")
                self.audio_available = False
    
    def _send_alert(self, alert: Alert) -> bool:
        if not self.audio_available:
            return False
        
        try:
            # 優先度に応じた音声パターン
            beep_patterns = {
                Priority.LOW: [1000, 100],      # 1回短いビープ
                Priority.MEDIUM: [1000, 200, 1000, 200],  # 2回ビープ
                Priority.HIGH: [1500, 300, 1500, 300, 1500, 300],  # 3回高音ビープ
                Priority.CRITICAL: [2000, 500, 2000, 500, 2000, 500, 2000, 500]  # 4回緊急音
            }
            
            pattern = beep_patterns.get(alert.priority, [1000, 200])
            
            # ビープ音生成と再生
            self._play_beep_pattern(pattern)
            
            logger.info(f"音声通知再生: {alert.priority.value}")
            return True
            
        except Exception as e:
            logger.error(f"音声通知エラー: {e}")
            return False
    
    def _play_beep_pattern(self, pattern: List[int]):
        """ビープ音パターン再生"""
        try:
            for i in range(0, len(pattern), 2):
                if i + 1 < len(pattern):
                    frequency = pattern[i]
                    duration = pattern[i + 1]
                    
                    # 簡易ビープ音（Windowsの場合）
                    if os.name == 'nt':
                        import winsound
                        winsound.Beep(frequency, duration)
                    else:
                        # Linux/Mac用の代替実装
                        print(f"\a")  # システムベル
                    
                    time.sleep(0.1)  # 音の間隔
                    
        except Exception as e:
            logger.error(f"ビープ音再生エラー: {e}")

class AlertNotificationSystem:
    """アラート・通知システム"""
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = Path(config_file)
        self.channels: List[NotificationChannel] = []
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Callable] = {}
        self.running = False
        self.monitor_thread = None
        
        # デフォルト設定
        self.default_config = {
            "desktop_notifications": True,
            "audio_notifications": True,
            "email_notifications": False,
            "slack_notifications": False,
            "email_config": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "to_emails": []
            },
            "slack_config": {
                "webhook_url": ""
            },
            "alert_rules": {
                "high_confidence_signals": True,
                "risk_violations": True,
                "large_price_movements": True,
                "system_errors": True,
                "performance_alerts": True
            },
            "thresholds": {
                "confidence_threshold": 0.8,
                "price_change_threshold": 0.05,
                "risk_score_threshold": 0.8
            }
        }
        
        self.load_config()
        self.setup_channels()
        
        logger.info("アラート・通知システム初期化完了")
    
    def load_config(self):
        """設定読み込み"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = self.default_config.copy()
                self.save_config()
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self):
        """設定保存"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")
    
    def setup_channels(self):
        """通知チャンネル設定"""
        self.channels = []
        
        # デスクトップ通知
        if self.config.get("desktop_notifications", True):
            self.channels.append(DesktopNotificationChannel())
        
        # 音声通知
        if self.config.get("audio_notifications", True):
            self.channels.append(AudioNotificationChannel())
        
        # メール通知
        if self.config.get("email_notifications", False):
            email_config = self.config.get("email_config", {})
            if email_config.get("username") and email_config.get("password"):
                self.channels.append(EmailNotificationChannel(
                    smtp_server=email_config.get("smtp_server", "smtp.gmail.com"),
                    smtp_port=email_config.get("smtp_port", 587),
                    username=email_config["username"],
                    password=email_config["password"],
                    to_emails=email_config.get("to_emails", []),
                    enabled=True
                ))
        
        # Slack通知
        if self.config.get("slack_notifications", False):
            slack_config = self.config.get("slack_config", {})
            if slack_config.get("webhook_url"):
                self.channels.append(SlackNotificationChannel(
                    webhook_url=slack_config["webhook_url"],
                    enabled=True
                ))
        
        logger.info(f"通知チャンネル設定完了: {len(self.channels)}チャンネル")
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """アラート送信"""
        results = {}
        
        # 履歴に保存
        self.alert_history.append(alert)
        
        # 履歴サイズ制限
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # 各チャンネルに送信
        for channel in self.channels:
            try:
                success = channel.send(alert)
                results[channel.name] = success
            except Exception as e:
                logger.error(f"チャンネル {channel.name} 送信エラー: {e}")
                results[channel.name] = False
        
        logger.info(f"アラート送信完了: {alert.title} - {results}")
        return results
    
    def create_trading_signal_alert(self, symbol: str, signal: str, probability: float, 
                                  confidence: float, price: float) -> Alert:
        """取引シグナルアラート作成"""
        priority = Priority.HIGH if confidence >= 0.8 else Priority.MEDIUM
        
        return Alert(
            id=f"signal_{symbol}_{int(time.time())}",
            alert_type=AlertType.TRADING_SIGNAL,
            priority=priority,
            title=f"{symbol} {signal}シグナル",
            message=f"{symbol}で{signal}シグナルを検出しました。確率: {probability:.1%}, 信頼度: {confidence:.1%}",
            symbol=symbol,
            data={
                "signal": signal,
                "probability": probability,
                "confidence": confidence,
                "price": price
            }
        )
    
    def create_risk_warning_alert(self, risk_type: str, current_value: float, 
                                limit_value: float, symbol: str = None) -> Alert:
        """リスク警告アラート作成"""
        severity_ratio = current_value / limit_value
        priority = Priority.CRITICAL if severity_ratio >= 1.5 else Priority.HIGH
        
        return Alert(
            id=f"risk_{risk_type}_{int(time.time())}",
            alert_type=AlertType.RISK_WARNING,
            priority=priority,
            title=f"リスク警告: {risk_type}",
            message=f"{risk_type}が制限値を超過しています。現在値: {current_value:.1%}, 制限値: {limit_value:.1%}",
            symbol=symbol,
            data={
                "risk_type": risk_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "severity_ratio": severity_ratio
            }
        )
    
    def create_price_alert(self, symbol: str, current_price: float, change_percent: float) -> Alert:
        """価格変動アラート作成"""
        abs_change = abs(change_percent)
        priority = Priority.HIGH if abs_change >= 0.1 else Priority.MEDIUM
        
        direction = "急騰" if change_percent > 0 else "急落"
        
        return Alert(
            id=f"price_{symbol}_{int(time.time())}",
            alert_type=AlertType.PRICE_ALERT,
            priority=priority,
            title=f"{symbol} {direction}",
            message=f"{symbol}が{change_percent:+.1%}の変動を記録しました。現在価格: ${current_price:,.2f}",
            symbol=symbol,
            data={
                "current_price": current_price,
                "change_percent": change_percent,
                "direction": direction
            }
        )
    
    def create_system_error_alert(self, error_type: str, error_message: str) -> Alert:
        """システムエラーアラート作成"""
        return Alert(
            id=f"error_{int(time.time())}",
            alert_type=AlertType.SYSTEM_ERROR,
            priority=Priority.CRITICAL,
            title=f"システムエラー: {error_type}",
            message=f"システムエラーが発生しました: {error_message}",
            data={
                "error_type": error_type,
                "error_message": error_message
            }
        )
    
    def create_performance_alert(self, metric_name: str, current_value: float, 
                               threshold: float, is_improvement: bool = True) -> Alert:
        """パフォーマンスアラート作成"""
        priority = Priority.MEDIUM if is_improvement else Priority.HIGH
        
        status = "改善" if is_improvement else "悪化"
        
        return Alert(
            id=f"perf_{metric_name}_{int(time.time())}",
            alert_type=AlertType.PERFORMANCE,
            priority=priority,
            title=f"パフォーマンス{status}: {metric_name}",
            message=f"{metric_name}が{status}しました。現在値: {current_value:.1%}, 基準値: {threshold:.1%}",
            data={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                "is_improvement": is_improvement
            }
        )
    
    def get_alert_history(self, hours: int = 24, alert_type: AlertType = None) -> List[Alert]:
        """アラート履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        if alert_type:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.alert_type == alert_type
            ]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_stats(self, hours: int = 24) -> Dict:
        """アラート統計取得"""
        recent_alerts = self.get_alert_history(hours)
        
        stats = {
            "total_alerts": len(recent_alerts),
            "by_type": {},
            "by_priority": {},
            "by_hour": {}
        }
        
        for alert in recent_alerts:
            # タイプ別統計
            alert_type = alert.alert_type.value
            stats["by_type"][alert_type] = stats["by_type"].get(alert_type, 0) + 1
            
            # 優先度別統計
            priority = alert.priority.value
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
            
            # 時間別統計
            hour = alert.timestamp.hour
            stats["by_hour"][hour] = stats["by_hour"].get(hour, 0) + 1
        
        return stats
    
    def update_config(self, new_config: Dict):
        """設定更新"""
        self.config.update(new_config)
        self.save_config()
        self.setup_channels()
        logger.info("アラート設定が更新されました")

# テスト関数
def test_alert_system():
    """アラートシステムのテスト"""
    print("=== アラート・通知システムテスト ===")
    
    # システム初期化
    alert_system = AlertNotificationSystem("test_alert_config.json")
    
    # テストアラート作成
    test_alerts = [
        alert_system.create_trading_signal_alert("BTC", "BUY", 0.75, 0.85, 67000),
        alert_system.create_risk_warning_alert("総エクスポージャー", 0.65, 0.6),
        alert_system.create_price_alert("ETH", 3200, 0.08),
        alert_system.create_system_error_alert("API接続", "Hyperliquid API接続タイムアウト"),
        alert_system.create_performance_alert("予測精度", 0.65, 0.6, True)
    ]
    
    # アラート送信テスト
    for alert in test_alerts:
        print(f"\nテストアラート送信: {alert.title}")
        results = alert_system.send_alert(alert)
        for channel, success in results.items():
            status = "成功" if success else "失敗"
            print(f"  {channel}: {status}")
        time.sleep(1)
    
    # 統計表示
    print(f"\nアラート統計:")
    stats = alert_system.get_alert_stats(1)
    print(f"総アラート数: {stats['total_alerts']}")
    print(f"タイプ別: {stats['by_type']}")
    print(f"優先度別: {stats['by_priority']}")

if __name__ == "__main__":
    test_alert_system()