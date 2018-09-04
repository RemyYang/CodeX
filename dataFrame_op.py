# coding: utf-8
import numpy as np
import pandas as pd
import click
import json
# define data of picture and model
global dataFram
#dataFram = pd.read_excel("./picData.xlsx", sheetname="sheet1")


'''测试通过'''
'''通过excel表格创建dataFram'''
@click.command()
@click.option('--datasrc', default='./picData.xlsx', help='the class of data')
def createDataFrameFromExcel(datasrc):
    dataFram = pd.read_excel(datasrc, sheetname = "sheet1")
    #print(dataFram)
    #print(dataGram.describe())
    saveDataFramToExcel(dataFram, "moduleDataNew.xlsx", "result")
    pass

'''仅仅用于测试'''
def createDataFramFromJson(jsonstr):
    data = json.loads(jsonstr)
    dataFram = pd.DataFrame(data,columns=['Name','Path','Probability','NoDish'],index=[])
    print(data)
    pass

'''仅仅用于测试'''
def createJsonStrFromDict(dict):
    json_str = json.dumps(dict)
    return json_str
    pass

'''测试通过'''
'''通过key,keyvalue来查询dataFram'''
def searchDataFromDataFramWithKeyAndValue(df,key,keyValue):
    df2 = df[df.get(key).isin([keyValue])]
    return df2
    pass

'''测试通过'''
'''通过key,keyvalue来查询非keyvalue的dataFram'''
def searchDataFromDataFramWithKeyAndNoValue(df,key,keyValue):
    df3 = df[df.get(key) != keyValue]
    #print(df3)
    return df3
    pass

'''测试通过'''
'''通过key,keyvalue来更新dataFram'''
def updateDataFromDataFramWithKeyAndValue(df,searchKey,searchKeyValue,updateKey,updateKeyValue):
    for ind in df[df[searchKey] == searchKeyValue].index.tolist():
        df.at[ind,updateKey] = updateKeyValue
    #print(df)
    pass

'''测试通过'''
'''通过key,keyvalue来查询非keyvalue的dataFram'''
def searchDataFromDataFramWithKeyAndBiggerValue(df,key,keyValue):
    df3 = df[df.get(key) >= keyValue]
    #print(df3)
    return df3
    pass

'''测试通过'''
'''导出dataFram到Excel'''
def saveDataFramToExcel(df,fileName,sheetName):
    writer = pd.ExcelWriter(fileName)
    df.to_excel(writer, sheetName)
    writer.save()
    pass

def saveDataFramToFile(df,fileName):
    data = {}
    foodlist= df.to_json(orient='records')
    data['datalist']=foodlist
    data['name']='bread'
    data['model_path']='/home/bread/modelpath'
    jsObj = json.dumps(data)
    f = file(fileName,'w')
    f.write(str(jsObj))
    f.close()

if __name__ == "__main__":
    #searchDataFromDataFramWithKeyAndValue(dataFram,"NoDish",-1)
    #searchDataFromDataFramWithKeyAndNoValue(dataFram,"NoDish",-1)
    #saveDataFramToFile(dataFram,"./dish.json")
    #createDataFrameFromExcel()
    #print(dataGram)
    #updateDataFromDataFramWithKeyAndValue(dataGram,"Name","bing_0002.jpg","Probability","0.78")
    '''
    dict = {
        'Name':'bing_0001.jpg',
        'Path':'//172.168.100.101/dataset/000001/bing_0001.jpg',
        'Probability':0.98,
        'NoDish':0,
        }
    jsonstr=createJsonStrFromDict(dict)
    createDataFramFromJson(jsonstr)
    '''