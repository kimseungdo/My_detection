import re
import time

prevtime = time.time()

HAN_CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
           'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


HAN_JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
            'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
            'ㅣ']


HAN_JONG = ['  ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
            'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

 
txt = input("낱자를 분석할 한글 완성자 문자열을 입력하세요: ")
#rtxt = re.sub(r'[^가-힣]', '', txt)  # 한글 완성자만 남김
retxt = ''

#딱스트 요소 하나씩 넘기는데 한글이면 변형 숫자면 패스 특문이면
for c in txt:
    
    if ord('가') <= ord(c) <=ord('힣'):
        cc = ord(c) - 44032     # 한글 완성자의 유니코드 포인터 값 추출
        cho = cc // (21 * 28)   # 초성 값 추출
        jung = (cc // 28) % 21  # 중성 값 추출
        jong = cc % 28          # 종성 값 추출
        
        #print("| %s | %s%s%s |" % (c, HAN_CHO[cho], HAN_JUNG[jung], HAN_JONG[jong] ), end="")
        #print(" 0x%s | %5d |" % ( hex(cc + 44032).upper()[-4:] , cc + 44032) )
        #print("           중값 %d 종값 %d : " %(jung, jong) )

        '''
        if jung == 2 or 6:
            jung = jung-2
        elif jung == 12 or 17:
            jung = jung-4
        '''
        if jung == 17:
            jung = jung-4
        if jung == 12:
            jung = jung-4
        if jung == 6:
            jung = jung-2
        if jung == 2:
            jung = jung-2
            
            jong = 0 #종값 삭제
                
        
        car_val = ((cho*21)+jung)*28+jong+0xAC00 #문자열 합쳐서 붙여버릴거
        retxt += chr(car_val) 
        
        #print("문자열 변환했다 야발놈들아 -- %s" %chr(car_val) )
            
        #end 한글
    elif ord('?') == ord(c): #? 9를 오인함 그래서 바꿔버릴거 
        c = str(9)
        retxt += c
            
    elif ord('0') <= ord(c) <= ord('9'): #정상 숫자면 저장
        retxt += str(c)
    #print("문자열 변환했다 야발놈들아 -- %s" %chr(car_val) )

print(retxt)
print("걸린시간이여 : %0.5f" %(time.time() - prevtime) )
'''
가나다라마 ㅑ2
너더러머버서어저 ㅕ6
고노도로모보소오조 ㅛ12
구누두루무부수우주 ㅠ17

아바사자

하허호

for a in rtxt:
    car_val = ((cho*21) + jung)*28+jong+0xAC00
    a += car_val

chr(a)
#print(type(car_val) )
#print(type(chr(car_val)) )
print("유니코드로 조합: %s" %a)
'''

