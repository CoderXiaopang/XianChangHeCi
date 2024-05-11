'''
@Project ：002_Code
@File    ：main.py
@IDE     ：PyCharm
@Author  ：Duangang Qu
@E-Mail  ：quduangang@outlook.com
@Date    ：2023/8/8 23:04
'''

import os
import re
import numpy as np
import pandas as pd
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from updated_untitled_ui_v2 import Ui_MainWindow

os.environ['QT_LOGGING_RULES'] = '*.debug=true'

class Worker(QThread):
    update_signal = pyqtSignal(str)  # 创建一个信号，用于更新GUI

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):

        self.update_signal.emit('-> 正在搜集综合数据处理信息....')
        df_simpleInfo = self.main_window.collect_data()
        self.update_signal.emit('-> 正在搜集样品单信息....')
        df_tupe = self.main_window.collect_type()
        result_df = pd.merge(df_simpleInfo, df_tupe[['样品编号', '类型']], on='样品编号', how='left')
        result_df = result_df.drop_duplicates(subset=['样品编号'], keep='first')
        self.update_signal.emit('-> 正在计算信号量....')
        single_df = self.main_window.caculate_single(result_df)
        result_df = pd.merge(result_df, single_df, on='样品编号', how='left')
        self.update_signal.emit('-> 计算参数....')
        result_df = self.main_window.caculate_args_value(result_df)
        self.update_signal.emit('-> 查找T2截止值....')
        result_df = self.main_window.caculate_T2(result_df)
        result_df['计算值-参考值(原样)'] = result_df['原样信号量（计算）'] - result_df['原样信号量']
        result_df['计算值-参考值(饱水)'] = result_df['饱水信号量（计算）'] - result_df['饱水信号量']
        result_df['计算值-参考值(饱锰)'] = result_df['饱锰信号量（计算）'] - result_df['饱锰信号量']

        cols_order = [
            "深度m", "样品编号", "原样信号量（计算）", "饱水信号量（计算）", "饱锰信号量（计算）",
            "原样质量g", "饱和质量g", "体积cm3", "密度(计算)g/cm3", "粘土信号量",
            "T2粘土截止值ms", "油体积cm3", "逸失体积cm3", "极限油孔隙度%", "粘土孔隙度%",
            "总孔隙度%", "有效孔隙度%", "地面总含油饱和度%", "地面总含水饱和度%", "逸失量%",
            "极限总含油饱和度%", "地面有效含油饱和度%", "地面极限有效含油饱和度%", "含油质量mg/g", "Tgm",
             "原样信号量", "计算值-参考值(原样)","饱水信号量","计算值-参考值(饱水)", "饱锰信号量","计算值-参考值(饱锰)","密度g/cm3", "密度(计算)-密度",
        ]

        from datetime import datetime
        current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result_{current_time_str}.xlsx'
        df_reordered = result_df[cols_order]
        df_reordered = df_reordered.sort_values(by='样品编号', key=lambda x: x.map(self.custom_sort))
        new_column_data = [ i+1 for i in range(len(df_reordered))]
        df_reordered.insert(1, '序号', new_column_data)
        df_reordered = df_reordered.rename(columns={'密度g/cm3': '密度（参考值）'})
        df_reordered = df_reordered.rename(columns={'密度(计算)g/cm3': '密度g/cm3'})
        df_reordered = df_reordered.rename(columns={'原样信号量': '原样信号量（参考值）'})
        df_reordered = df_reordered.rename(columns={'饱水信号量': '饱水信号量（参考值）'})
        df_reordered = df_reordered.rename(columns={'饱锰信号量': '饱锰信号量（参考值）'})
        df_reordered = df_reordered.rename(columns={'密度(计算)-密度': '计算值-参考值（密度）'})


        df_reordered.to_excel(filename,index=False)
        self.update_signal.emit('计算已完成，请查看文件！')
        self.update_signal.emit(f'{filename}')

    def custom_sort(self,item):
        # 使用 '-' 分割编号
        parts = item.split('-')

        # 返回一个元组，第一个元素是'-'前的数字，第二个元素是'-'后的数字
        return (int(parts[0]), int(parts[1]))

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.start_worker)

    def start_worker(self):
        self.worker = Worker(self)  # 创建 Worker 线程并传入 MyWindow 的引用
        self.worker.update_signal.connect(self.update_gui)  # 连接 Worker 的信号到 GUI 更新函数
        self.worker.start()  # 开始 Worker 线程

    def update_gui(self, message):
        self.textEdit.append(message)  # 更新 GUI


    def by_caculate_pore(self, row, water_k, oil_k, oil_density,water_b,oil_b):

        return (((row['饱水信号量（计算）'] - row['饱锰信号量（计算）'] - water_b) / water_k) + (((row['饱锰信号量（计算）'] - oil_b) / oil_k )/ oil_density) / row['体积cm3'] )* 100

    def noemal_caculate_pore(self, row, water_k, oil_k, oil_density,water_b,oil_b):
        # =(   (E2-F2)/17885+F2/22245/0.8111   )/I2*100
        return ((row['原样信号量（计算）'] - row['饱锰信号量（计算）']-water_b) / water_k + (
                    row['饱水信号量（计算）'] + row['饱锰信号量（计算）'] - row['原样信号量（计算）'] - oil_b) / oil_k / oil_density) / row['体积cm3'] * 100

    # def by_caculate_ysl(self, row, water_k, oil_k, oil_density,water_b,oil_b):
    #
    #     return ((row['饱水信号量（计算）'] - row['饱锰信号量（计算）'] - water_b) / water_k + (row['饱锰信号量（计算）'] - oil_b) / oil_k / oil_density) / row['体积cm3'] * 100
    #
    # def noemal_caculate_ysl(self, row, water_k, oil_k, oil_density,water_b,oil_b):
    #     # =(   (E2-F2)/17885+F2/22245/0.8111   )/I2*100
    #     return ((row['原样信号量（计算）'] - row['饱锰信号量（计算）'] - water_b) / water_k + (
    #                 row['饱水信号量（计算）'] + row['饱锰信号量（计算）'] - row['原样信号量（计算）'] - oil_b) / oil_k / oil_density) / row['体积cm3'] * 100

    def caculate_args_value(self, df):
        oil_k = float(self.lineEdit_2.text())
        water_k = float(self.lineEdit.text())
        water_b = float(self.lineEdit_7.text())
        oil_b = float(self.lineEdit_6.text())
        oil_density = float(self.lineEdit_3.text())
        df['油体积cm3'] = ((df['饱锰信号量（计算）'] - oil_b) / oil_k )/ oil_density
        df['逸失体积cm3'] = df.apply(lambda row:0 if row['类型'] == '保压' else ((row['饱水信号量（计算）'] - row['原样信号量（计算）'] - water_b) / water_k) if row['逸失体积cm3'] == -1 else row['逸失体积cm3'], axis=1)
        df['极限油孔隙度%'] = ((df['油体积cm3'] + df['逸失体积cm3']) / df['体积cm3']) * 100
        df['总孔隙度%'] = df.apply(lambda row: self.by_caculate_pore(row, water_k, oil_k, oil_density,water_b,oil_b) if row['类型'] == '保压' else self.noemal_caculate_pore(row, water_k, oil_k, oil_density,water_b,oil_b), axis=1)
        df['地面总含油饱和度%'] = ((df['油体积cm3'] / df['体积cm3'] )/ df['总孔隙度%']) * 10000
        df['逸失量%'] = df.apply(lambda row:0 if row['类型'] == '保压' else(((row['逸失体积cm3'] / row['体积cm3'] )/ row['总孔隙度%']) * 10000 if row['逸失量%'] == -1 else row['逸失量%']),axis=1)
        df['地面总含水饱和度%'] = 100 - df['地面总含油饱和度%'] - df['逸失量%']
        df['极限总含油饱和度%'] = df['地面总含油饱和度%'] + df['逸失量%']
        df['含油质量mg/g'] = ((df['油体积cm3'] * oil_density) / df['原样质量g']) * 1000
        df['密度(计算)g/cm3'] = df['饱和质量g'] /df['体积cm3']
        df['密度(计算)-密度'] = df['密度(计算)g/cm3'] -df['密度g/cm3']

        return df

    def caculate_T2(self, df):
        required_columns = ['粘土孔隙度%', '地面有效含油饱和度%', '地面极限有效含油饱和度%', 'T2', '粘土信号量','有效孔隙度%']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        arr = []
        for index, row in df.iterrows():
            t2 = 0
            if row['总孔隙度%'] >= 13:
                if row['密度g/cm3'] < 2.5:
                    t2 = 45.6065 * row['密度g/cm3'] ** 2 - 217.7654 * row['密度g/cm3'] + 260.3357
                else:
                    t2 = 40.657 * row['密度g/cm3'] ** 2 - 203.61 * row['密度g/cm3'] + 255.78
            elif row['密度g/cm3'] < 2.4:
                t2 = 19.814 * row['密度g/cm3'] ** 2 - 91.463 * row['密度g/cm3'] + 105.88
            elif row['总孔隙度%'] < 9:
                t2 = -26.2206 * row['密度g/cm3'] ** 2 + 132.5388 * row['密度g/cm3'] - 166.8826
                if t2 < 0.14:
                    t2 = 0.14
            elif 9 <= row['总孔隙度%'] <= 10:
                if 2.5 < row['密度g/cm3'] < 2.6:
                    t2 = 281.8853 * row['密度g/cm3'] ** 2 -1432.16 * row['密度g/cm3'] + 1819.513
                else:
                    if row['Tgm'] < 0.9:
                        t2  = (1.6775 * row['Tgm'] * row['Tgm'] - 2.1502 * row['Tgm'] + 1.1272)
                    else:
                        t2 = -26.2206 * row['密度g/cm3'] ** 2 + 132.5388 * row['密度g/cm3'] -166.8826
            elif 10<row['总孔隙度%'] <13:
                if row['Tgm'] < 0.69:
                    df.loc[index, 'Tgm'] = 0.69
                    t2 = 2.0524 * row['总孔隙度%'] + 30.9511 * row['Tgm'] - 0.0854* row['总孔隙度%'] ** 2 -18.7147*0.69*0.69 -24.2552
                elif row['Tgm'] > 0.9:
                    t2 = 3.704 * row['Tgm'] * row['Tgm'] - 8.2882 * row['Tgm'] + 5.16
                else:
                    t2 = 2.0524 * row['总孔隙度%'] + 30.9511 * row['Tgm'] - 0.0854* row['总孔隙度%'] ** 2-18.7147*row['Tgm']*row['Tgm'] -24.2552
            # 这里已经获得T2了
            t2, Clay = self.t2_and_Clay(row['饱水文件地址'],t2)
            df.at[index, '粘土孔隙度%'] = (Clay/float(self.lineEdit.text())/row['体积cm3']) * 100

            pore_row = row['总孔隙度%']- df.at[index,'粘土孔隙度%']
            if (row['总孔隙度%'] - df.at[index, '粘土孔隙度%'] - row['极限油孔隙度%']) < 0.1:
                df.at[index, '有效孔隙度%'] = df.at[index,'极限油孔隙度%'] + (df.at[index,'总孔隙度%'] - df.at[index,'极限油孔隙度%']) * 0.1
            else:
                df.at[index, '有效孔隙度%'] = (df.at[index,'总孔隙度%'] - df.at[index,'粘土孔隙度%'])


            df.at[index, '地面有效含油饱和度%'] = ((df.at[index,'油体积cm3']/df.at[index,'体积cm3'])/ df.at[index,'有效孔隙度%'])*10000
            df.at[index, '地面极限有效含油饱和度%'] = (((df.at[index,'油体积cm3'] + df.at[index,'逸失体积cm3']) / df.at[index,'体积cm3']) / df.at[index,'有效孔隙度%']) * 10000

            if pore_row - row['极限油孔隙度%'] < 0.1:
                df.loc[index, '有效孔隙度%'] = df.at[index,'极限油孔隙度%']  +(df.at[index,'总孔隙度%']  - df.at[index,'极限油孔隙度%'] )*0.1
                df.loc[index, '粘土孔隙度%'] = df.at[index,'总孔隙度%'] - df.at[index,'有效孔隙度%']
                Clay = (df.at[index,'粘土孔隙度%']* (float(self.lineEdit.text())/df.at[index,'体积cm3']))/100
                t2 = self.tttt2(df.at[index,'饱水文件地址'],Clay)
            df.at[index, 'T2粘土截止值ms'] = t2
            df.at[index, '粘土信号量'] = Clay
        return df



    def tttt2(self,path,value):
        df = pd.read_csv(path, sep="\s+", header=None)
        df.columns = ['A', 'B']

        df['C'] = df['B'].cumsum()

        # 找到与value差值最小的C列的索引
        closest_index = abs(df['C'] - value).idxmin()

        # 返回该索引对应的A列的值
        return df.loc[closest_index, 'A']



    def t2_and_Clay(self,path,value):
        data = pd.read_csv(path, sep="\s+", header=None)
        data.columns = ['A', 'B']
        diffs = abs(data['A'] - value)

        min_index = diffs.idxmin()

        # 获取该索引对应的值
        closest_value = data.loc[min_index, 'A']

        # 计算前index行B列的和
        sum_B = data.loc[:min_index, 'B'].sum()

        return closest_value,  sum_B







    def caculate_single(self, result_df):
        '''
        计算出所有样本的信号量，如果 原样 > 保水  直接对调
        :param result_df:
        :return:
        '''

        path = self.lineEdit_4.text()
        result = []
        # 遍历文件夹
        for root, dirs, files in os.walk(path):
            for file_name in files:
                if file_name.startswith('~'):
                    continue
                if file_name.endswith(('.Inv', '.inv')):
                    result.append(os.path.join(root, file_name))
        simple_num_list = result_df['样品编号'].values
        arr_all = []

        for simple_name in simple_num_list:
            arr_temp = [0, 0, 0, 0,0,0,-1,-1]
            for path in result:
                path_name = self.extract_number(os.path.basename(path))
                if len(path_name) < 1:
                    pass
                else:
                    if simple_name == str(path_name[0]):
                        arr_temp[3] = simple_name
                        tgm_path = ''
                        if '原样' in path:
                            tgm_path = path.replace("原样", "饱水")
                            arr_temp[0] = self.caculate_singke_value(path)
                            arr_temp[1] = self.caculate_singke_value(path.replace("原样", "饱水"))
                            arr_temp[2] = self.caculate_singke_value(path.replace("原样", "饱锰"))
                        if '饱水' in path:
                            tgm_path = path
                            arr_temp[0] = self.caculate_singke_value(path.replace("饱水", "原样"))
                            arr_temp[1] = self.caculate_singke_value(path)
                            arr_temp[2] = self.caculate_singke_value(path.replace("饱水", "饱锰"))
                        if '饱锰' in path:
                            tgm_path = path.replace("饱锰", "饱水")
                            arr_temp[0] = self.caculate_singke_value(path.replace("饱锰", "原样"))
                            arr_temp[1] = self.caculate_singke_value(path.replace("饱锰", "饱水"))
                            arr_temp[2] = self.caculate_singke_value(path)
                        if arr_temp[0] - arr_temp[1] > 0:
                            arr_temp[6] = 0
                            arr_temp[7] = 0
                            temp = arr_temp[1]
                            arr_temp[1] = arr_temp[0]
                            arr_temp[0] = temp
                            path_water = path.replace("原样", "饱水").replace("饱锰", "饱水")  # 饱水文件地址
                            path_simple = path_water.replace("饱水", "原样")
                            self.swap_filenames(path_water, path_simple)
                        tgm = self.caculate_tgm(tgm_path,arr_temp[1])
                        arr_temp[4] =tgm
                        arr_temp[5] = tgm_path
                        arr_all.append(arr_temp)
                        break

        df = pd.DataFrame(arr_all)
        df.columns = ['原样信号量（计算）', '饱水信号量（计算）', '饱锰信号量（计算）', '样品编号','Tgm','饱水文件地址','逸失量%','逸失体积cm3']
        return df

    def caculate_tgm(self,path,value):
        data = pd.read_csv(path, sep="\s+", header=None)
        data.columns = ['A', 'B']
        data['C'] = data['A'] ** (data['B']/value)
        return np.product(data['C'])






    def caculate_singke_value(self, path):
        data = pd.read_csv(path, sep="\s+", header=None)
        data.columns = ['A', 'B']
        return np.sum(data['B'])

    def swap_filenames(self, path_A, path_B):
        '''
        对调两个文件的文件名称
        :param path_B:
        :return:
        '''
        # 生成一个临时文件名
        temp_path = path_A + ".temp"

        # 交换文件名
        os.rename(path_A, temp_path)
        os.rename(path_B, path_A)
        os.rename(temp_path, path_B)

    def extract_number(self, data):
        return re.findall(r'-(\d+-\d+)-', data)

    def collect_data(self):
        path = self.lineEdit_4.text()
        result = []
        # 遍历文件夹
        for root, dirs, files in os.walk(path):
            for file_name in files:
                if file_name.startswith('~'):
                    continue
                if "综合数据处理" in file_name and file_name.endswith(('.xls', '.xlsx')):
                    result.append(os.path.join(root, file_name))

        selected_columns = ['样品编号', '深度m', '原样信号量', '饱水信号量', '饱锰信号量', '原样质量g', '饱和质量g',
                            '体积cm3', '密度g/cm3']
        all_data = []  # 用于存储所有选中的数据
        for path_file in result:
            try:
                data = pd.read_excel(path_file, skiprows=2)
                selected_data = data[selected_columns]
                all_data.append(selected_data)
            except Exception as e:
                print(f"在处理文件 {path_file} 时发生错误: {str(e)}")
                continue

        df = pd.concat(all_data, ignore_index=True)

        return df

    def collect_type(self):
        result = []
        path = self.lineEdit_4.text()
        for root, dirs, files in os.walk(path):
            for file_name in files:
                if file_name.startswith('~'):
                    continue
                if "实验样品单" in file_name and file_name.endswith(('.xls', '.xlsx')):
                    result.append(os.path.join(root, file_name))
        selected_columns = ['样号', '类型']
        all_data = []  # 用于存储所有选中的数据
        for path_file in result:
            try:
                # 读取 Excel 文件的所有 sheet 名称
                xls = pd.ExcelFile(path_file)
                # 取最后一个 sheet 名称
                sheet_name = xls.sheet_names[-1]

                # 读取表格的前两行作为标题
                header_row1 = pd.read_excel(path_file, sheet_name=sheet_name, header=None, nrows=1).values.tolist()[0]
                header_row2 = \
                    pd.read_excel(path_file, sheet_name=sheet_name, header=None, skiprows=1, nrows=1).values.tolist()[0]
                merged_header = [str(a) + str(b) if b is not None else str(a) for a, b in zip(header_row1, header_row2)]
                merged_header = [x.replace('nan', "") for x in merged_header]

                # 重新读取表格，使用合并后的标题
                data = pd.read_excel(path_file, sheet_name=sheet_name, header=None, skiprows=2, names=merged_header)
                selected_data = data[selected_columns]
                all_data.append(selected_data)
            except Exception as e:
                print(f"在处理文件 {path_file} 时发生错误: {str(e)}")
                continue

        df = pd.concat(all_data, ignore_index=True)
        df.columns = ['样品编号', '类型']
        return df


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
