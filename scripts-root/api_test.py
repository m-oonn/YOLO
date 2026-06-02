import requests
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

class APITester:
    def __init__(self):
        self.results = []
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def test_endpoint(self, method, endpoint, name, data=None, params=None):
        """测试单个 API 端点"""
        url = f"{BASE_URL}{endpoint}"
        result = {
            'name': name,
            'method': method,
            'endpoint': endpoint,
            'status_code': None,
            'response_time': None,
            'success': False,
            'response': None,
            'error': None
        }

        start_time = time.time()
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=10)
            else:
                result['error'] = f'Unsupported method: {method}'
                self.results.append(result)
                return result

            result['response_time'] = (time.time() - start_time) * 1000
            result['status_code'] = response.status_code

            try:
                result['response'] = response.json()
            except:
                result['response'] = response.text

            result['success'] = response.status_code == 200

        except requests.exceptions.ConnectionError:
            result['error'] = '连接失败 - 服务未启动'
            result['response_time'] = (time.time() - start_time) * 1000
        except requests.exceptions.Timeout:
            result['error'] = '请求超时'
            result['response_time'] = (time.time() - start_time) * 1000
        except Exception as e:
            result['error'] = str(e)
            result['response_time'] = (time.time() - start_time) * 1000

        self.results.append(result)
        return result

    def print_result(self, result):
        """打印单个测试结果"""
        print(f"\n{'='*70}")
        print(f"📌 {result['name']}")
        print(f"   方法: {result['method']} {result['endpoint']}")
        print(f"{'='*70}")

        if result['error']:
            print(f"   ❌ 错误: {result['error']}")
            return

        print(f"   状态码: {result['status_code']}")
        print(f"   响应时间: {result['response_time']:.2f}ms")

        if result['success']:
            print(f"   ✅ 成功")
        else:
            print(f"   ❌ 失败")

        print(f"\n   响应内容:")
        response_str = str(result['response'])
        if len(response_str) > 500:
            response_str = response_str[:500] + "...(截断)"
        print(f"   {response_str}")

    def print_summary(self):
        """打印测试汇总"""
        total = len(self.results)
        success = sum(1 for r in self.results if r['success'])
        failed = total - success

        print(f"\n\n{'#'*70}")
        print(f"#                    测试报告汇总")
        print(f"{'#'*70}")
        print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 总测试数: {total}")
        print(f"# 成功: {success} ✅")
        print(f"# 失败: {failed} ❌")
        print(f"# 成功率: {(success/total*100):.1f}%")
        print(f"{'#'*70}\n")

        print(f"\n{'='*70}")
        print(f"详细结果:")
        print(f"{'='*70}")

        for i, result in enumerate(self.results, 1):
            status_icon = "✅" if result['success'] else "❌"
            status_text = "正常" if result['success'] else "异常"

            if result['error']:
                print(f"{i}. {status_icon} {result['name']:<30} | {status_text}")
                print(f"   错误: {result['error']}")
            else:
                print(f"{i}. {status_icon} {result['name']:<30} | {status_text}")
                print(f"   状态码: {result['status_code']} | 响应时间: {result['response_time']:.2f}ms")

    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'#'*70}")
        print(f"#      YOLO 课程设计项目 - 后端 API 全面测试")
        print(f"# 测试目标: {BASE_URL}")
        print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        print("\n🔄 开始测试...\n")

        # 1. 健康检查
        self.test_endpoint('GET', '/health', '健康检查端点')

        # 2. 摄像头 API
        self.test_endpoint('GET', '/api/cameras/', '获取摄像头列表')

        # 3. 检测 API - 状态
        self.test_endpoint('GET', '/api/detection/status', '获取检测状态')

        # 4. 检测 API - 进度（重点测试）
        self.test_endpoint('GET', '/api/detection/progress', '获取检测进度')

        # 5. 检测 API - 开始检测
        self.test_endpoint('POST', '/api/detection/start', '开始检测', data={'camera_id': 'default'})

        # 等待一小段时间再停止
        time.sleep(0.5)

        # 6. 检测 API - 停止检测
        self.test_endpoint('POST', '/api/detection/stop', '停止检测')

        # 7. 事件 API - 统计
        self.test_endpoint('GET', '/api/events/stats', '获取事件统计')

        # 8. 事件 API - 列表
        self.test_endpoint('GET', '/api/events/', '获取事件列表')

        # 9. 告警 API - 统计
        self.test_endpoint('GET', '/api/alarms/stats', '获取告警统计')

        # 10. 告警 API - 列表
        self.test_endpoint('GET', '/api/alarms/', '获取告警列表')

        # 11. 配置 API
        self.test_endpoint('GET', '/api/config', '获取系统配置')

        # 12. MLLM API - 状态
        self.test_endpoint('GET', '/api/mllm/status', '获取 MLLM 状态')

        # 打印所有结果
        for result in self.results:
            self.print_result(result)

        # 打印汇总
        self.print_summary()

        return self.results

if __name__ == '__main__':
    tester = APITester()
    results = tester.run_all_tests()
