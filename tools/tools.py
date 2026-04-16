import os
import heapq
import re
import os
import json
# import ijson
# from rdflib import Graph



# def get_ttl(input_file, out_file):
#     cnt = 0    
#     # 创建一个图对象
#     g = Graph()
#     # 解析 .ttl 文件
#     try:
#         g.parse(input_file, format="ttl")
#     except Exception as e:
#         print(f"Error parsing file: {e}")

#     with open(out_file, "w", encoding="utf-8") as outfile:
#         # 遍历图中的所有三元组
#         for subj, pred, obj in g.triples((None, None, None)):
#             print(f"Subject: {subj}, Predicate: {pred}, Object: {obj}")
#             previous_item = None

#             cnt += 1       
#             # 写入输出文件
#             outfile.write(f"<{subj}>\t<{pred}>\t<{obj}>\n")    
#             # 输出处理进度，每处理 10,000 个对象输出一次
#             progress = cnt + 1
#             if progress % 10000 == 0:
#                 print(f"处理进度: {progress}")



def remove_text_in_brackets(input_file, output_file, strong_clean_flag = False):
    def process(line):
        """  
        删除字符串中括号及其中的内容。
        包括括号 () 和中文括号（）。
        :param line: 原始字符串
        :return: 删除括号内容后的字符串
        """
        # 使用正则表达式匹配括号及其中的内容
        # 匹配圆括号 () 或中文括号（），以及其中的内容
        return re.sub(r"[\(\（][^\)\）]*[\)\）]", "", line)

    def remove_non_alpha(string):
        # 使用正则表达式删除首尾非字母字符
        cleaned_string = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', string)
        return cleaned_string

    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在！")
        return

    try:
        # 打开输入文件和输出文件
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            cnt = 0
            # 按行读取输入文件
            for line in infile:
                cnt += 1       
                # 删除当前行中括号及其中的内容
                line_wo_bracket = process(line)
                if strong_clean_flag == True:
                    # 删除前后的数字和符号
                    processed_line = remove_non_alpha(line_wo_bracket)
                # 写入到输出文件
                outfile.write(processed_line + "\n")
                if cnt % 10000 == 0:
                    print(f"处理进度: {cnt}")

        print(f"处理完成！结果已保存到 {output_file}")
    except Exception as e:
        print(f"处理文件时出现错误：{e}")


def sort_large_file(
    input_file, output_file, chunk_size=100000
):  # chunk_size 可根据内存调整
    chunks = []
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            while True:
                lines = infile.readlines(chunk_size)
                if not lines:
                    break
                lines.sort()  # 对每一块数据进行排序

                chunk_filename = f"temp_chunk_{len(chunks)}.txt"
                chunks.append(chunk_filename)
                with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
                    chunk_file.writelines(lines)

        # 合并排序后的块
        cnt = 1
        with open(output_file, "w", encoding="utf-8") as outfile:
            files = [open(f, "r", encoding="utf-8") for f in chunks]
           
            for line in heapq.merge(*files):
                cnt += 1
                outfile.write(line)
                if cnt % 10000 == 0:
                    print(f'merging process: {cnt}/{len(chunks)*chunk_size}')
            for f in files:
                f.close()

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 清理临时文件
        for chunk in chunks:
            try:
                os.remove(chunk)
            except FileNotFoundError:
                pass


def remove_duplicates(input_file, output_file):
    """
    从已排序的文本文件中去除重复行。

    Args:
        input_file: 已排序的输入文件路径。
        output_file: 输出无重复行的文件路径。
    """
    try:
        duplicate_cnt = 0
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            # 初始化前一行的内容
            previous_line = None
            index = 0
            for current_line in infile:
                index += 1
                # 如果当前行与前一行不同，则写入输出文件
                if current_line != previous_line:
                    outfile.write(current_line)
                    duplicate_cnt += 1
                    previous_line = current_line
                progress = index + 1
                if progress % 10000 == 0:
                    print(f"处理进度: {progress}")
        
        print(f"去除重复 {duplicate_cnt} 行完成")

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 不存在。")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")


def str_full_to_half_width(input_file, output_file):
    def process(ustring):
        result = []
        for char in ustring:
            # 全角空格直接转换为半角空格
            if char == "\u3000":
                result.append(" ")
            # 处理其他全角字符（根据Unicode范围）
            elif "\uff01" <= char <= "\uff5e":
                # 将全角字符的Unicode码减去偏移量0xFEE0得到半角字符
                result.append(chr(ord(char) - 0xFEE0))
            else:
                # 保留原本是半角的字符
                result.append(char)
        return "".join(result)

    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在！")
        return

    try:
        # 打开输入文件和输出文件
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:

            # 按行读取输入文件
            for line in infile:
                # 对当前行的内容进行全角转半角处理
                converted_line = process(line)
                # 写入到输出文件
                outfile.write(converted_line)
        print(f"处理完成！转换后的内容已保存到 {output_file}")
    except Exception as e:
        print(f"处理文件时出现错误：{e}")

def file_split(input_file, num):
        # 假设你的文件名是 'data.txt'
        # input_file = 'input_file.txt'
        input_name = input_file.split('.')[0]
        file_type = input_file.split('.')[1]
        output_files = []
        for i in range(num):
            output_files.append(f'{input_name}_part{i}.{file_type}')

        # 计算文件总行数
        with open(input_file, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file)

        # 每个部分的行数
        lines_per_part = total_lines // 4

        # 逐行读取并写入
        with open(input_file, 'r', encoding='utf-8') as file:
            current_part = 0
            current_line = 0
            output_file = open(output_files[current_part], 'w', encoding='utf-8')
            
            for line in file:
                output_file.write(line)
                current_line += 1
                
                # 检查是否需要切换到下一个文件
                if current_line >= lines_per_part and current_part < 3:
                    output_file.close()
                    current_part += 1
                    output_file = open(output_files[current_part], 'w', encoding='utf-8')
                    current_line = 0
            
            # 关闭最后一个文件
            output_file.close()

def print_file(infile, n=10):
    # 打开文件并读取前10行
    with open(infile, 'r') as file:
        for i in range(n):
            line = file.readline()
            if line:  # 检查是否还有行可读
                print(line.strip())
            else:
                break  # 如果文件行数少于10行，提前退出

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        line_count = 0
        for line in file:
            line_count += 1
            if line_count % 1000000 == 0:
                print(line_count)

    print(f"文件 {filename} 中的行数为: {line_count}")


# def json_pass(infilename, outfilename):
#     with open(infilename, "r", encoding="utf-8") as infile, open(outfilename, "w", encoding="utf-8") as outfile:
#         for index, item in enumerate(ijson.items(infile, 'item')):  # 假设顶层是一个数组
#             try:                
#                 # 提取"en"字段的内容
#                 extracted = {
#                     "type": item.get("type", {}),
#                     "id": item.get("id", {}),
#                     "labels": item.get("labels", {}).get("en", ""),
#                     "descriptions": item.get("descriptions", {}).get("en", ""),
#                     "aliases": item.get("aliases", {}).get("en", [])
#                 }
                
#                 # 将提取结果写入输出文件
#                 outfile.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            
#                 # 输出处理进度，每处理 10,000 个对象输出一次
#                 progress = index + 1
#                 if progress % 10000 == 0:
#                     print(f"处理进度: {progress}")
#             except:
#                 pass

#     print(f"提取完成，结果已保存到 {outfile}")

def get_entities_from_triples(infilename, outfilename, signpass=[]):
    cnt = 0
    with open(infilename, 'r', encoding='utf-8') as infile, open(outfilename, 'w', encoding='utf-8') as outfile:
        # 初始化前一行的内容
        previous_item = None
        for line in infile:
            cnt += 1
            # 分割每行，提取第一列
            current_item = line.split("\t")[0]
            # 过滤掉特殊符号
            for sign in signpass:
                current_item = current_item.strip(sign)
            current_item = re.sub(r"_+", " ", current_item)
            if current_item != previous_item:
                # 写入输出文件
                outfile.write(current_item + '\n')    
                previous_item = current_item      
            # 输出处理进度，每处理 10,000 个对象输出一次
            progress = cnt + 1
            if progress % 10000 == 0:
                print(f"处理进度: {progress}")
def get_entities_from_wikidata(infilename, outfilename1, outfilename2, outfilename3):
    with open(infilename, "r", encoding="utf-8") as infile, open(outfilename1, "w", encoding="utf-8") as outfile1, open(outfilename2, "w", encoding="utf-8") as outfile2, open(outfilename3, "w", encoding="utf-8") as outfile3:
        index = 0
        for line in infile:  # 假设顶层是一个数组try:  
            index += 1    
            item = json.loads(line)
            # 提取"en"字段的内容
            try:
                entity = item.get("labels", {}).get("value", "")
                aliases_dic = item.get("aliases", [])
                aliases = []
                for item in aliases_dic:
                    aliases.append(item.get("value", ""))
                descriptions = item.get("descriptions", "")
                if descriptions != "":
                    descriptions = descriptions.get("value", "")
            except:
                pass
                            
            # 将提取结果写入输出文件
            outfile1.write(entity + "\n")
            outfile2.write(entity + "\n")
            for item in aliases:
                outfile2.write(item + "\n")
            outfile3.write(f"<{entity}>\t<{aliases}>\t<{descriptions}>\n")
        
            # 输出处理进度，每处理 10,000 个对象输出一次
            progress = index + 1
            if progress % 10000 == 0:
                print(f"处理进度: {progress}")

def main():
    # infile = '/data/H-RAG/H_RAG/data/kg/wikidata-20240101-all.json'
    # outfile = '/data/H-RAG/H_RAG/data/kg/extracted.json'
    # data1 = "/data/H-RAG/H_RAG/data/kg/triples.txt"
    # 示例用法
    # input_path = "CN-DBpediaMention2Entity.txt"  # 替换成你的已排序输入文件路径
    # temp1 = "temp1.txt"
    # temp2 = "temp2.txt"
    # output_path = "CN-DBpedia_pass.txt"  # 替换成你想要输出的文件路径
    # remove_duplicates('triple_data/wiki_entities_sorted.txt', output_file = 'triple_data/wiki_entities.txt')
    # remove_duplicates('triple_data/wiki_aliases_sorted.txt', output_file = 'triple_data/wiki_aliases.txt')
    # remove_duplicates('triple_data/wiki_descriptions_sorted.txt', output_file = 'triple_data/wiki_descriptions.txt')
    # str_full_to_half_width(temp1, temp2)
    # remove_duplicates(temp2, output_path)
    # file_split('CN-DBpedia_pass.txt', 4)
    # print_file('/data/H-RAG/H_RAG/data/kg/triples.txt', 1000)
    # count_lines('/data/H-RAG/H_RAG/data/kg/triples.txt')
    # get_entities_from_triples("triples.txt", "triples_entities.txt", ['>', '<'])
    # get_entities_from_wikidata('/data/H-RAG/H_RAG/data/kg/extracted.json', 'wiki_entities.txt', 'wiki_aliases.txt', 'wiki_descriptions.txt')
    # yago-beyond-wikipedia, yago-facts, yago-schema, yago-taxonomy, yago-schema
    # get_ttl('/data/H-RAG/H_RAG/data/kg/yago-schema.ttl', 'yago_wiki_triples.txt')
    # json_pass(infile, outfile)
    # remove_text_in_brackets('triple_data/dbpedia_entities.txt', 'triple_data/dbpedia_entities_clean.txt', True)
    remove_duplicates('triple_data/temp.txt', 'triple_data/dbpedia_entities_clean.txt')


if __name__ == "__main__":
    main()
