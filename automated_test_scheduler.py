#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動テストスケジューラー
バックグラウンドで継続的にテストを実行
"""

import schedule
import time
import threading
import asyncio
from datetime import datetime, timedelta
import json
import subprocess
import sys
import os
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.automated_paper_testing import AutomatedPaperTestingSystem
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedTestScheduler:
    """自動テストスケジューラー"""
    
    def __init__(self):
        self.is_running = False
        self.test_system = None
        self.current_test_thread = None
        self.test_status = {
            'is_active': False,
            'start_time': None,
            'current_day': 0,
            'total_days': 7,
            'strategies_running': []
        }
        
        # 結果保存ディレクトリ
        self.results_dir = Path("automated_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def start_scheduler(self):
        """スケジューラー開始"""
        logger.info("自動テストスケジューラー開始")
        self.is_running = True
        
        # 即座にテスト開始
        self.start_test()
        
        # 定期実行スケジュール
        schedule.every(5).minutes.do(self.check_test_status)
        schedule.every(1).hours.do(self.save_intermediate_results)
        schedule.every().day.at("00:00").do(self.daily_report)
        
        # スケジューラーループ
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # 1分ごとにチェック
    
    def start_test(self):
        """テスト開始"""
        if self.test_status['is_active']:
            logger.warning("テストは既に実行中です")
            return
        
        logger.info("新しい自動テストを開始します")
        
        # 非同期テストを別スレッドで実行
        self.current_test_thread = threading.Thread(
            target=self._run_async_test,
            daemon=True
        )
        self.current_test_thread.start()
        
        # ステータス更新
        self.test_status.update({
            'is_active': True,
            'start_time': datetime.now(),
            'current_day': 0,
            'strategies_running': [s.value for s in TradingStrategy]
        })
    
    def _run_async_test(self):
        """非同期テストの実行"""
        try:
            # 新しいイベントループを作成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # テストシステム初期化
            self.test_system = AutomatedPaperTestingSystem(initial_balance=10000)
            
            # テスト実行
            result = loop.run_until_complete(
                self.test_system.start_automated_test(duration_days=7)
            )
            
            # 結果保存
            self._save_final_results(result)
            
            # ステータスリセット
            self.test_status['is_active'] = False
            
            logger.info("自動テスト完了")
            
        except Exception as e:
            logger.error(f"テスト実行エラー: {e}")
            self.test_status['is_active'] = False
        finally:
            loop.close()
    
    def check_test_status(self):
        """テストステータスのチェック"""
        if not self.test_status['is_active']:
            return
        
        # 経過日数計算
        if self.test_status['start_time']:
            elapsed = datetime.now() - self.test_status['start_time']
            self.test_status['current_day'] = elapsed.days + 1
            
            logger.info(f"テスト進行中: {self.test_status['current_day']}/{self.test_status['total_days']}日目")
    
    def save_intermediate_results(self):
        """中間結果の保存"""
        if not self.test_system or not self.test_status['is_active']:
            return
        
        try:
            # 現在のパフォーマンスを取得
            intermediate_report = {
                'timestamp': datetime.now().isoformat(),
                'day': self.test_status['current_day'],
                'strategies': {}
            }
            
            for strategy in TradingStrategy:
                metrics = self.test_system._calculate_performance_metrics(strategy)
                intermediate_report['strategies'][strategy.value] = metrics
            
            # ファイル保存
            filename = self.results_dir / f"intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(intermediate_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"中間結果保存: {filename}")
            
        except Exception as e:
            logger.error(f"中間結果保存エラー: {e}")
    
    def daily_report(self):
        """日次レポート生成"""
        if not self.test_status['is_active']:
            return
        
        logger.info(f"日次レポート生成 - Day {self.test_status['current_day']}")
        
        # Emailやチャットへの通知をここに実装可能
        # 現在はログ出力のみ
    
    def _save_final_results(self, result: dict):
        """最終結果の保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON形式で保存
        json_file = self.results_dir / f"final_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # サマリーテキスト生成
        summary_file = self.results_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("自動ペーパートレーディングテスト結果サマリー\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"テスト期間: {result['test_duration']}\n")
            f.write(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("戦略別パフォーマンス:\n")
            for strategy, metrics in result['strategies'].items():
                f.write(f"\n{strategy.upper()}:\n")
                f.write(f"  総取引数: {metrics['total_trades']}\n")
                f.write(f"  勝率: {metrics['win_rate']:.1f}%\n")
                f.write(f"  総損益: ${metrics['total_pnl']:,.2f}\n")
                f.write(f"  最大ドローダウン: {metrics['max_drawdown']*100:.1f}%\n")
                f.write(f"  シャープレシオ: {metrics['sharpe_ratio']:.2f}\n")
            
            f.write(f"\n最優秀戦略: {result['best_strategy']}\n")
            f.write(f"最高利益: ${result['summary']['best_total_pnl']:,.2f}\n")
        
        logger.info(f"最終結果保存完了: {json_file}")
    
    def stop_scheduler(self):
        """スケジューラー停止"""
        logger.info("スケジューラー停止中...")
        self.is_running = False
        
        # 実行中のテストがあれば待機
        if self.current_test_thread and self.current_test_thread.is_alive():
            logger.info("実行中のテスト完了を待機中...")
            self.current_test_thread.join(timeout=300)  # 最大5分待機

def main():
    """メイン関数"""
    scheduler = AutomatedTestScheduler()
    
    try:
        print("🤖 自動ペーパートレーディングテストスケジューラー")
        print("="*60)
        print("テストが自動的に開始され、1週間継続実行されます。")
        print("中間結果は1時間ごとに保存されます。")
        print("Ctrl+C で停止できます。")
        print("="*60)
        
        scheduler.start_scheduler()
        
    except KeyboardInterrupt:
        print("\n停止シグナルを受信しました...")
        scheduler.stop_scheduler()
        print("スケジューラーを正常に停止しました。")

if __name__ == "__main__":
    # 必要なインポート
    from core.automated_paper_testing import TradingStrategy
    
    main()