#!/usr/bin/env python3
"""
进度管理器 - 类似Docker下载镜像的并行任务进度显示
"""

import threading
import time
import queue
from typing import Dict, Optional, Any
from rich.console import Console
from rich.live import Live
from rich.table import Table
from enum import Enum
import multiprocessing as mp
from dataclasses import dataclass


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    total_steps: Optional[int] = None
    current_step_num: int = 0
    error_msg: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ProgressManager:
    """
    并行任务进度管理器，类似Docker镜像下载的进度显示
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.tasks: Dict[str, TaskInfo] = {}
        self.lock = threading.Lock()
        self.is_running = False
        self._display_thread: Optional[threading.Thread] = None
        self._update_queue: queue.Queue = queue.Queue()

    def add_task(
        self, task_id: str, name: str, total_steps: Optional[int] = None
    ) -> None:
        """添加一个新任务"""
        with self.lock:
            self.tasks[task_id] = TaskInfo(
                task_id=task_id, name=name, total_steps=total_steps
            )

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        current_step_num: Optional[int] = None,
        error_msg: Optional[str] = None,
    ) -> None:
        """更新任务状态"""
        try:
            self._update_queue.put(
                {
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "current_step": current_step,
                    "current_step_num": current_step_num,
                    "error_msg": error_msg,
                    "timestamp": time.time(),
                },
                block=False,
            )
        except queue.Full:
            pass  # 如果队列满了，跳过这次更新

    def _apply_update(self, update: Dict[str, Any]) -> None:
        """应用更新到任务信息"""
        task_id = update["task_id"]

        with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]

            if update["status"] is not None:
                old_status = task.status
                task.status = update["status"]

                # 记录状态变化时间
                if (
                    old_status == TaskStatus.PENDING
                    and task.status == TaskStatus.RUNNING
                ):
                    task.start_time = update["timestamp"]
                elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task.end_time = update["timestamp"]

            if update["progress"] is not None:
                task.progress = min(100.0, max(0.0, update["progress"]))

            if update["current_step"] is not None:
                task.current_step = update["current_step"]

            if update["current_step_num"] is not None:
                task.current_step_num = update["current_step_num"]

            if update["error_msg"] is not None:
                task.error_msg = update["error_msg"]

    def _create_display_table(self) -> Table:
        """创建显示表格"""
        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("Task", style="cyan", width=25)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=50)
        table.add_column("Current Step", style="dim", width=30)

        with self.lock:
            for task in self.tasks.values():
                # 状态显示
                if task.status == TaskStatus.PENDING:
                    status_text = "[yellow]⏳ Pending[/yellow]"
                elif task.status == TaskStatus.RUNNING:
                    status_text = "[blue]🔄 Running[/blue]"
                elif task.status == TaskStatus.COMPLETED:
                    status_text = "[green]✅ Done[/green]"
                elif task.status == TaskStatus.FAILED:
                    status_text = "[red]❌ Failed[/red]"
                else:
                    status_text = "[dim]❓ Unknown[/dim]"

                # 进度条
                if task.status == TaskStatus.COMPLETED:
                    progress_text = "[green]█████████████████████████████████████████████████[/green] 100%"
                elif task.status == TaskStatus.FAILED:
                    progress_text = f"[red]{'█' * int(task.progress / 2)}{'░' * (50 - int(task.progress / 2))}[/red] {task.progress:.1f}%"
                elif task.status == TaskStatus.RUNNING:
                    filled = int(task.progress / 2)  # 50个字符的进度条
                    progress_text = f"[blue]{'█' * filled}{'░' * (50 - filled)}[/blue] {task.progress:.1f}%"
                else:
                    progress_text = "[dim]░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[/dim] 0%"

                # 当前步骤
                current_step_text = task.current_step or ""
                if task.total_steps and task.current_step_num > 0:
                    current_step_text = f"[{task.current_step_num}/{task.total_steps}] {current_step_text}"

                # 如果有错误，显示错误信息
                if task.error_msg:
                    current_step_text = f"[red]Error: {task.error_msg}[/red]"

                table.add_row(
                    task.name[:24] + "..." if len(task.name) > 24 else task.name,
                    status_text,
                    progress_text,
                    current_step_text[:29] + "..."
                    if len(current_step_text) > 29
                    else current_step_text,
                )

        return table

    def _display_loop(self) -> None:
        """显示循环，在单独线程中运行"""
        with Live(
            self._create_display_table(), refresh_per_second=4, console=self.console
        ) as live:
            while self.is_running:
                try:
                    # 处理所有待更新的任务
                    while True:
                        try:
                            update = self._update_queue.get_nowait()
                            self._apply_update(update)
                        except queue.Empty:
                            break

                    # 更新显示
                    live.update(self._create_display_table())
                    time.sleep(0.25)

                except Exception:
                    # 忽略显示错误，继续运行
                    pass

    def start(self) -> None:
        """开始显示进度"""
        if self.is_running:
            return

        self.is_running = True
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def stop(self) -> None:
        """停止显示进度"""
        self.is_running = False
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)

    def get_summary(self) -> Dict[str, int]:
        """获取任务摘要统计"""
        summary = {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0}

        with self.lock:
            for task in self.tasks.values():
                summary["total"] += 1
                if task.status == TaskStatus.PENDING:
                    summary["pending"] += 1
                elif task.status == TaskStatus.RUNNING:
                    summary["running"] += 1
                elif task.status == TaskStatus.COMPLETED:
                    summary["completed"] += 1
                elif task.status == TaskStatus.FAILED:
                    summary["failed"] += 1

        return summary

    def is_all_completed(self) -> bool:
        """检查是否所有任务都已完成"""
        with self.lock:
            return all(
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                for task in self.tasks.values()
            )

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """等待所有任务完成"""
        start_time = time.time()

        while not self.is_all_completed():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)

        return True


# 全局进度管理器实例（用于多进程通信）
_global_progress_manager: Optional[ProgressManager] = None
_manager_queue: Optional[mp.Queue] = None


def init_global_progress_manager(console: Optional[Console] = None) -> ProgressManager:
    """初始化全局进度管理器"""
    global _global_progress_manager, _manager_queue

    _global_progress_manager = ProgressManager(console)
    _manager_queue = mp.Queue()

    return _global_progress_manager


def get_global_progress_manager() -> Optional[ProgressManager]:
    """获取全局进度管理器"""
    return _global_progress_manager


def update_task_progress(task_id: str, **kwargs) -> None:
    """更新任务进度（可在子进程中调用）"""
    global _global_progress_manager

    if _global_progress_manager:
        _global_progress_manager.update_task(task_id, **kwargs)


# 装饰器，用于自动跟踪函数执行进度
def track_progress(task_id: str, task_name: str, total_steps: Optional[int] = None):
    """
    装饰器：自动跟踪函数执行进度

    Args:
        task_id: 任务ID
        task_name: 任务显示名称
        total_steps: 总步骤数
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 添加任务
            if _global_progress_manager:
                _global_progress_manager.add_task(task_id, task_name, total_steps)
                _global_progress_manager.update_task(
                    task_id, status=TaskStatus.RUNNING, progress=0.0
                )

            try:
                result = func(*args, **kwargs)

                # 标记完成
                if _global_progress_manager:
                    _global_progress_manager.update_task(
                        task_id,
                        status=TaskStatus.COMPLETED,
                        progress=100.0,
                        current_step="Completed",
                    )

                return result

            except Exception as e:
                # 标记失败
                if _global_progress_manager:
                    _global_progress_manager.update_task(
                        task_id, status=TaskStatus.FAILED, error_msg=str(e)
                    )
                raise

        return wrapper

    return decorator
