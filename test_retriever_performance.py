#!/usr/bin/env python3
"""
Retriever Server 高级性能测试脚本
专门针对Dense/BM25检索服务的并发和性能测试
"""

import asyncio
import aiohttp
import time
import statistics
import argparse
import psutil
import GPUtil
import json
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


class RetrieverAdvancedTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.retrieve_url = f"{base_url}/retrieve"
        self.system_stats = []
        self.monitoring = False
        
    def start_system_monitoring(self):
        """启动系统资源监控"""
        self.monitoring = True
        self.system_stats = []
        
        def monitor():
            while self.monitoring:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU监控
                gpu_stats = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_stats.append({
                            'id': gpu.id,
                            'load': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                        })
                except:
                    gpu_stats = []
                
                self.system_stats.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'gpu_stats': gpu_stats
                })
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_system_monitoring(self):
        """停止系统资源监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)

    async def single_request(self, session: aiohttp.ClientSession, queries: List[str], 
                           topk: int = 3, return_scores: bool = True, timeout: int = 30) -> Dict[str, Any]:
        """发送单个检索请求"""
        payload = {
            "queries": queries,
            "topk": topk,
            "return_scores": return_scores
        }
        
        start_time = time.time()
        try:
            async with session.post(
                self.retrieve_url, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    total_docs = sum(len(query_result) for query_result in result.get("result", []))
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "query_count": len(queries),
                        "total_docs": total_docs,
                        "avg_docs_per_query": total_docs / len(queries) if queries else 0,
                        "error": None
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "query_count": len(queries),
                        "total_docs": 0,
                        "avg_docs_per_query": 0,
                        "error": f"HTTP {response.status}: {error_text[:200]}"
                    }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "status_code": 0,
                "query_count": len(queries),
                "total_docs": 0,
                "avg_docs_per_query": 0,
                "error": str(e)
            }

    async def test_different_batch_sizes(self, base_queries: List[str], 
                                       batch_sizes: List[int] = [1, 2, 5, 10, 20],
                                       concurrent_requests: int = 5) -> Dict[str, Any]:
        """测试不同批量大小的性能"""
        print(f"\n🔄 测试不同批量大小的性能 (并发数: {concurrent_requests})")
        print("=" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 测试批量大小: {batch_size}")
            
            # 创建测试查询批次
            query_batches = []
            for i in range(concurrent_requests * 2):  # 每个并发连接发送2个请求
                batch = base_queries[:batch_size] * (batch_size // len(base_queries) + 1)
                query_batches.append(batch[:batch_size])
            
            # 执行并发测试
            connector = aiohttp.TCPConnector(limit=concurrent_requests)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    self.single_request(session, queries, topk=3, return_scores=True)
                    for queries in query_batches
                ]
                
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
            
            # 统计结果
            successful = [r for r in responses if isinstance(r, dict) and r.get("success")]
            failed = [r for r in responses if not (isinstance(r, dict) and r.get("success"))]
            
            if successful:
                total_time = end_time - start_time
                total_queries = sum(r["query_count"] for r in successful)
                avg_response_time = statistics.mean(r["response_time"] for r in successful)
                qps = total_queries / total_time
                
                results[batch_size] = {
                    "batch_size": batch_size,
                    "success_count": len(successful),
                    "fail_count": len(failed),
                    "total_queries": total_queries,
                    "total_time": total_time,
                    "avg_response_time": avg_response_time,
                    "qps": qps,
                    "queries_per_second": qps
                }
                
                print(f"   ✅ 成功: {len(successful)}/{len(responses)}")
                print(f"   📊 QPS: {qps:.2f}")
                print(f"   ⏱️  平均响应时间: {avg_response_time:.3f}s")
            else:
                print(f"   ❌ 所有请求失败")
                results[batch_size] = {"batch_size": batch_size, "qps": 0, "error": "all_failed"}
        
        return results

    async def test_concurrent_load(self, queries: List[str], 
                                 max_concurrent: int = 50, 
                                 step: int = 5,
                                 duration: int = 30) -> Dict[str, Any]:
        """测试并发负载能力"""
        print(f"\n🚀 测试并发负载能力 (持续时间: {duration}秒)")
        print("=" * 60)
        
        self.start_system_monitoring()
        
        results = {}
        
        for concurrent in range(step, max_concurrent + 1, step):
            print(f"\n🔄 测试并发数: {concurrent}")
            
            # 创建并发任务
            async def sustained_requests():
                connector = aiohttp.TCPConnector(limit=concurrent)
                async with aiohttp.ClientSession(connector=connector) as session:
                    end_time = time.time() + duration
                    responses = []
                    
                    while time.time() < end_time:
                        tasks = [
                            self.single_request(session, queries, topk=3)
                            for _ in range(concurrent)
                        ]
                        batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
                        responses.extend(batch_responses)
                        
                        # 短暂休息避免过载
                        await asyncio.sleep(0.1)
                    
                    return responses
            
            start_time = time.time()
            all_responses = await sustained_requests()
            total_time = time.time() - start_time
            
            # 统计结果
            successful = [r for r in all_responses if isinstance(r, dict) and r.get("success")]
            failed = [r for r in all_responses if not (isinstance(r, dict) and r.get("success"))]
            
            if successful:
                response_times = [r["response_time"] for r in successful]
                success_rate = len(successful) / len(all_responses) * 100
                qps = len(successful) / total_time
                
                results[concurrent] = {
                    "concurrent": concurrent,
                    "total_requests": len(all_responses),
                    "success_count": len(successful),
                    "success_rate": success_rate,
                    "qps": qps,
                    "avg_response_time": statistics.mean(response_times),
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                    "p99_response_time": statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
                }
                
                print(f"   ✅ 成功率: {success_rate:.1f}% ({len(successful)}/{len(all_responses)})")
                print(f"   📊 QPS: {qps:.2f}")
                print(f"   ⏱️  P95响应时间: {results[concurrent]['p95_response_time']:.3f}s")
                
                # 如果成功率太低，停止测试
                if success_rate < 70:
                    print(f"   ⚠️  成功率过低，停止测试")
                    break
            else:
                print(f"   ❌ 所有请求失败")
                break
            
            await asyncio.sleep(2)  # 让系统恢复
        
        self.stop_system_monitoring()
        return results

    def analyze_system_stats(self):
        """分析系统资源使用情况"""
        if not self.system_stats:
            return {}
        
        cpu_usage = [stat['cpu_percent'] for stat in self.system_stats]
        memory_usage = [stat['memory_percent'] for stat in self.system_stats]
        
        analysis = {
            "cpu": {
                "avg": statistics.mean(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory": {
                "avg": statistics.mean(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            }
        }
        
        # GPU统计
        if self.system_stats[0]['gpu_stats']:
            gpu_loads = []
            gpu_memory = []
            for stat in self.system_stats:
                if stat['gpu_stats']:
                    gpu_loads.extend([gpu['load'] for gpu in stat['gpu_stats']])
                    gpu_memory.extend([gpu['memory_percent'] for gpu in stat['gpu_stats']])
            
            if gpu_loads:
                analysis["gpu"] = {
                    "load_avg": statistics.mean(gpu_loads),
                    "load_max": max(gpu_loads),
                    "memory_avg": statistics.mean(gpu_memory),
                    "memory_max": max(gpu_memory)
                }
        
        return analysis

    def generate_report(self, batch_results: Dict, concurrent_results: Dict):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("📊 **RETRIEVER 性能测试报告**")
        print("=" * 80)
        
        # 批量大小测试结果
        if batch_results:
            print(f"\n🔸 **批量大小测试结果**:")
            best_batch_qps = 0
            best_batch_size = 1
            
            for batch_size, result in batch_results.items():
                if isinstance(result, dict) and "qps" in result:
                    qps = result["qps"]
                    print(f"   批量大小 {batch_size:2d}: QPS {qps:6.2f}, 响应时间 {result.get('avg_response_time', 0):.3f}s")
                    if qps > best_batch_qps:
                        best_batch_qps = qps
                        best_batch_size = batch_size
            
            print(f"   🏆 最佳批量大小: {best_batch_size} (QPS: {best_batch_qps:.2f})")
        
        # 并发测试结果
        if concurrent_results:
            print(f"\n🔸 **并发负载测试结果**:")
            best_concurrent_qps = 0
            best_concurrent = 1
            
            for concurrent, result in concurrent_results.items():
                qps = result["qps"]
                success_rate = result["success_rate"]
                p95_time = result["p95_response_time"]
                print(f"   并发数 {concurrent:2d}: QPS {qps:6.2f}, 成功率 {success_rate:5.1f}%, P95 {p95_time:.3f}s")
                
                if qps > best_concurrent_qps and success_rate >= 95:
                    best_concurrent_qps = qps
                    best_concurrent = concurrent
            
            print(f"   🏆 最佳并发数: {best_concurrent} (QPS: {best_concurrent_qps:.2f})")
        
        # 系统资源使用
        system_analysis = self.analyze_system_stats()
        if system_analysis:
            print(f"\n🔸 **系统资源使用情况**:")
            print(f"   CPU 使用率: 平均 {system_analysis['cpu']['avg']:.1f}%, 峰值 {system_analysis['cpu']['max']:.1f}%")
            print(f"   内存使用率: 平均 {system_analysis['memory']['avg']:.1f}%, 峰值 {system_analysis['memory']['max']:.1f}%")
            
            if "gpu" in system_analysis:
                gpu = system_analysis["gpu"]
                print(f"   GPU 负载: 平均 {gpu['load_avg']:.1f}%, 峰值 {gpu['load_max']:.1f}%")
                print(f"   GPU 内存: 平均 {gpu['memory_avg']:.1f}%, 峰值 {gpu['memory_max']:.1f}%")
        
        # 配置建议
        print(f"\n🔸 **配置建议**:")
        if batch_results and concurrent_results:
            best_batch = max(batch_results.keys(), key=lambda k: batch_results[k].get("qps", 0))
            best_concurrent = max(concurrent_results.keys(), key=lambda k: concurrent_results[k]["qps"])
            
            # 计算建议的配置
            suggested_rate_limit = min(best_concurrent, 20)  # 不超过20
            suggested_num_workers = min(suggested_rate_limit + 5, 25)  # worker略多于rate_limit
            
            print(f"   推荐批量大小: {best_batch}")
            print(f"   推荐配置参数:")
            print(f"     rate_limit = {suggested_rate_limit}")
            print(f"     num_workers = {suggested_num_workers}")
            print(f"     retrieval_batch_size = {best_batch}  # 在retriever server中设置")


async def main():
    parser = argparse.ArgumentParser(description="Retriever Server 高级性能测试")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Retriever服务地址")
    parser.add_argument("--max-concurrent", type=int, default=300, help="最大并发数")
    parser.add_argument("--test-duration", type=int, default=30, help="负载测试持续时间(秒)")
    parser.add_argument("--queries-file", help="自定义查询文件(JSON格式)")
    
    args = parser.parse_args()
    
    # 默认测试查询
    test_queries = [
        "什么是人工智能和机器学习？",
        "深度学习神经网络的基本原理",
        "自然语言处理技术在现实中的应用",
        "计算机视觉和图像识别技术发展",
        "强化学习算法的工作机制"
    ]
    
    # 如果提供了查询文件，加载自定义查询
    if args.queries_file:
        try:
            with open(args.queries_file, 'r', encoding='utf-8') as f:
                custom_queries = json.load(f)
                if isinstance(custom_queries, list):
                    test_queries = custom_queries
        except Exception as e:
            print(f"⚠️  无法加载查询文件: {e}, 使用默认查询")
    
    tester = RetrieverAdvancedTester(args.url)
    
    print("🚀 启动 Retriever Server 高级性能测试")
    print(f"📍 服务地址: {args.url}/retrieve")
    print(f"📋 测试查询数量: {len(test_queries)}")
    
    try:
        # 1. 测试不同批量大小
        batch_results = await tester.test_different_batch_sizes(
            test_queries, 
            batch_sizes=[1, 2, 3, 5, len(test_queries)],
            concurrent_requests=5
        )
        
        # 2. 测试并发负载
        concurrent_results = await tester.test_concurrent_load(
            test_queries,
            max_concurrent=args.max_concurrent,
            step=50,
            duration=args.test_duration
        )
        
        # 3. 生成报告
        tester.generate_report(batch_results, concurrent_results)
        
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
    finally:
        tester.stop_system_monitoring()


if __name__ == "__main__":
    asyncio.run(main()) 