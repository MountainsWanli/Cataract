import pandas as pd

def merge_captions(input_file, output_file):
    # 从CSV文件读取数据
    df = pd.read_csv(input_file)
    
    # 转换为浮点数
    df['start_sec'] = df['start_sec'].astype(float)
    df['end_sec'] = df['end_sec'].astype(float)
    
    # 初始化合并结果列表
    merged = []
    
    # 处理第一条记录
    if not df.empty:
        first_row = df.iloc[0]
        merged.append({
            'start_sec': first_row['start_sec'],
            'end_sec': first_row['end_sec'],
            'caption': first_row['caption'].strip()
        })
    
    # 处理剩余记录
    for i in range(1, len(df)):
        current = df.iloc[i]
        current_caption = current['caption'].strip()
        current_duration = current['end_sec'] - current['start_sec']
        
        # 获取上一条记录
        last = merged[-1]
        last_caption = last['caption'].strip().lower()
        current_caption_lower = current_caption.lower()
        
        # 规则1: 如果是Idle，归为上一条非Idle的caption
        if current_caption_lower == 'idle':
            last['end_sec'] = current['end_sec']
        # 规则2: 如果不是Idle且持续时间<=3秒，归为上一条caption
        elif current_duration <= 2:
            last['end_sec'] = current['end_sec']
        # 规则3: 如果下一条caption与上一条相同，合并时间
        elif current_caption_lower == last_caption:
            last['end_sec'] = current['end_sec']
        # 否则作为新的记录
        else:
            merged.append({
                'start_sec': current['start_sec'],
                'end_sec': current['end_sec'],
                'caption': current_caption
            })
    
    # 转换为DataFrame并格式化
    result_df = pd.DataFrame(merged)
    # 保留整数
    result_df['start_sec'] = result_df['start_sec'].round().astype(int)
    result_df['end_sec'] = result_df['end_sec'].round().astype(int)
    # 标题首字母大写，统一格式
    result_df['caption'] = result_df['caption'].apply(
        lambda x: x[0].upper() + x[1:].lower() if x and len(x) > 0 else x
    )
    # 去除多余空格（例如"2. 5 mm"改为"2.5 mm"）
    result_df['caption'] = result_df['caption'].str.replace('  ', ' ')
    
    # 保存到CSV文件
    result_df.to_csv(output_file, index=False)
    print(f"合并完成，结果已保存到 {output_file}")
    
    return result_df

# 示例用法
if __name__ == "__main__":
    # 输入CSV文件路径
    input_csv = "/home/itaer2/zxy/cataract/code/output/train2/Phaco-6.csv"
    # 输出CSV文件路径
    output_csv = "/home/itaer2/zxy/cataract/code/output/train2/Phaco-6_2.csv"
    
    # 执行合并操作
    merged_df = merge_captions(input_csv, output_csv)
    
    # 打印前几行结果查看
    print("\n合并后的前5条记录:")
    print(merged_df.head())
    