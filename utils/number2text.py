import re
import random

def num2txt(n):
    def _readNumber(n):
    
        units = [''] + list('십백천')
        nums = '일이삼사오육칠팔구'
        result = []
        i = 0
        while n > 0:
            n, r = divmod(n, 10)
            if r > 0:
              result.append(nums[r-1] + units[i])
            i += 1
        return ''.join(result[::-1])
    """1억미만의 숫자를 읽는 함수"""
    a, b = [_readNumber(x) for x in divmod(n, 10000)]
    if a:
        return a + "만" +  b
    return b


def dec2txt(strNum):
    
    # 만 단위 자릿수
    tenThousandPos = 4
    # 억 단위 자릿수
    hundredMillionPos = 9
    txtDigit = ['', '십', '백', '천', '만', '억']
    txtNumber = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    txtPoint = '점'
    
    resultStr = ''
    digitCount = 0
    
    # print(strNum)
    #자릿수 카운트
    for ch in strNum:
        # ',' 무시
        if ch == ',':
            continue
        #소숫점 까지
        elif ch == '.':
            break
        digitCount = digitCount + 1


    digitCount = digitCount-1
    index = 0

    while True:
        notShowDigit = False
        ch = strNum[index]
        #print(str(index) + ' ' + ch + ' ' +str(digitCount))
        # ',' 무시
        if ch == ',':
            index = index + 1
            if index >= len(strNum):
                break;
            continue

        if ch == '.':
            resultStr = resultStr + txtPoint
        else:
            #자릿수가 2자리이고 1이면 '일'은 표시 안함.
            # 단 '만' '억'에서는 표시 함
            if(digitCount > 1) and (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos) and int(ch) == 1:
                resultStr = resultStr + ''
            elif int(ch) == 0:
                resultStr = resultStr + ''
                # 단 '만' '억'에서는 표시 함
                if (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos):
                    notShowDigit = True
            else:
                resultStr = resultStr + txtNumber[int(ch)]


        # 1억 이상
        if digitCount > hundredMillionPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-hundredMillionPos]
        # 1만 이상
        elif digitCount > tenThousandPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-tenThousandPos]
        else:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount]

        if digitCount <= 0:
            digitCount = 0
        else:
            digitCount = digitCount - 1
        index = index + 1
        if index >= len(strNum):
            break;
    # print(resultStr)
    return resultStr


def time2txt(n):
    n = int(n)
    time_dict= {1:"한", 2:"두", 3:"세", 4:"네", 5:"다섯", 6:"여섯", 7:"일곱", 
                8:"여덟", 9:"아홉", 10:"열", 11:"열한", 12:"열두", 13:"열세", 
                14:"열네", 15:"열다섯", 16:"열여섯",17:"열일곱", 18:"열여덟", 19:"열아홉"}
    return time_dict[n]


def num2text(text):
    p=random.random()
        # first match decimal numbers
    decimal = re.findall("\d+\.\d+", text)
    if len(decimal)>0:
        for dec in decimal:
            dec2str = dec2txt(dec)
            if str(dec[0])=='0': # 점사 -> 영점사
                dec2str = '영'+dec2str
            if float(dec) > 10 and str(dec[0])=='1': # 일십일점오 -> 십일점오
                dec2str = dec2str[1:]    
            text = text.replace(dec, dec2str, 1)

    # math time where 10 should be read "열 " rather than "십 "
    time = re.findall("\d+\시", text)
    if len(time)>0:
        for t in time:
            if 20 > int(t[:-1]) > 0:
                time2str = time2txt(t[:-1])
                text = text.replace(t, time2str+'시',1)

    # match non-decimal numbers
    number = re.findall("\d+", text)
    if len(number)>0:
        for n in number:
            int_n = int(n)
            try:
                num2str = num2txt(int_n)
                if int_n >= 10 and num2str[0]=='일': # 일십구 -> 십구
                    num2str = num2str[1:]
                text = text.replace(n, num2str, 1)
            except:
                continue
    
    if p > 0.5:
        text = text.replace('%','프로')
    else:
        text= text.replace('%','퍼센트')
        
    return text