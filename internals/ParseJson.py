import json

def getParsedData(data):
    jsonObj = json.loads(data)
    classLabel = jsonObj['classLabel']
    print(classLabel)
    jsonEntity = jsonObj['data']
    print(len(jsonEntity))
    headerCount = 0
    header = ""
    parsedStr = ""    
    for entity in jsonEntity:
        count = 0
        print(len(entity))
        for prop in entity:
            if(headerCount == 0):
                header = header + prop
            parsedStr = parsedStr + entity[prop]
            if(count < len(entity)-1):
                parsedStr = parsedStr + ','
                if(headerCount == 0):
                    header = header + ','
            count += 1
        parsedStr = parsedStr + "\n"
        headerCount += 1
    parsedStr = header + "\n" + parsedStr
    print(parsedStr)
    
    f = open('data.csv', 'w')
    f.write(parsedStr)
    f.close()            
    return (classLabel, parsedStr)

if __name__ == '__main__':
    data = '{"classLabel" : "one single column name","data": [{"BookTitle": "Leading","BookID": "56353","BookAuthor": "Sir Alex Ferguson"},{"BookTitle": "How Google Works","BookID": "73638","BookAuthor": "Eric Smith"},{"BookTitle": "The Merchant of Venice","BookID": "37364","BookAuthor": "William Shakespeare"}]}'
    parsedData = getParsedData(data)
    print(parsedData[0])
    print(parsedData[1])
    