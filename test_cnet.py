import socket
import struct

a,b = 0, 1
def calculate_checksum(data):
    # 计算字节流的校验和
    return sum(data) & 0xFF

def unpack(rev_data):
    # 解析数据包(文件头、类型、命令、url源类型、校验)i
    header, type_, cmdid, source_cls, check = struct.unpack('BBBBB', rev_data[:5])

    # 计算校验和
    calculated_checksum = calculate_checksum(rev_data[:-1])

    # 检查校验位是否准确，校验不正确返回-1，正确返回cmdid, source_cls
    if check == calculated_checksum:
        return cmdid, source_cls
    else:
        return -1, -1


class DetecResultQ:
    def __init__(self,results=[], m=0):
        self.header = 0xff 
        self.type = 0x000000aa    
        self.m = m  # 识别数量
        self.results = results# 识别结果，最多五个目标
        self.check = 0x00

    def sort_results_by_prob(self):
        if self.m > 5:
            # 按照 results[i].prob 从大到小排序
            sorted_results = sorted(self.results, key=lambda result: result.prob, reverse=True)
            # 保留前五个对象
            self.results = sorted_results[:5]  
            self.m = 5  
    
    # def calculate_checksum(self, data):
    #     # 计算字节流的校验和
    #     return sum(data) & 0xFF
    
    def pack(self):
        # 使用struct.pack指定每个属性的大小和格式
        # '10s' 表示一个长度为10的字符串，'i' 表示一个整数
        packed_header = struct.pack('B', self.header)
        packed_type = struct.pack('i', self.type)
        packed_m = struct.pack('i', self.m)
        # 使用循环将每个类对象的属性打包
        packed_results = b""
        for result in self.results:
            packed_results += struct.pack('iiiiif', result.id, result.x, result.y, result.w, result.h, result.prob)

        self.check = calculate_checksum(packed_header + packed_type + packed_m + packed_results)
        packed_check = struct.pack('B', self.check)
        return packed_header + packed_type + packed_m + packed_results + packed_check
    
    def unpack(self,data):
        # 使用struct.unpack解析二进制数据
        header, type, m = struct.unpack('Bii', data[:9])
        
        # 解析每个类对象的属性
        results = []
        offset = 9
        for _ in range(self.m):
            result_data = struct.unpack('iiiiif', data[offset:offset+24])
            results.append(result_data)
            offset += 24

        check = struct.unpack('B', data[-1:])
        return header, type, m, results, check

class DetectResult:   
    def __init__(self, id, x, y, w, h, prob):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.prob = prob

result = DetectResult(0, 10, 10, 400, 100, 0.65)
results=[]
for i in range(10):
    results.append(result)
resultQ = DetecResultQ(results=results,m=len(results))  
resultQ.sort_results_by_prob()
data = resultQ.pack()

ip_port = ("192.168.8.200", 20005)
soc = socket.socket()
soc.connect(ip_port)

print('发送信息:\n')
while True:
    # soc.sendall(a.encode('utf-8'))
    soc.sendall(data)
    reply = soc.recv(5)
    cmdid, source_cls = unpack(reply)
    a =1


soc.close()       # 关闭连接