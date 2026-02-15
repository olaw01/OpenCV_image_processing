import cv2
import imutils

img = cv2.imread(r'C:\Users\awawr\PythonProject\OpenCV_image_processing\01_basics\assets\view.jpg')
logo = cv2.imread(r'C:\Users\awawr\PythonProject\OpenCV_image_processing\01_basics\assets\python.png')
logo = imutils.resize(logo, height=150)

# cv2.imshow('img', img)
# cv2.imshow('logo', logo)
# cv2.waitKey(0)

# wycięcie obszaru roi - region of interest (rozmiar zdjecia lgo) z obrazu view
rows, cols, channels = logo.shape
roi = img[:rows, :cols]
# cv2.imshow('roi', roi)
# cv2.waitKey(0)

# przekształcenie logo do odcieni szarości
gray = cv2.cvtColor(src=logo, code=cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

# budowanie maski zawierajacej logo i tekst
mask = cv2.threshold(src=gray, thresh=220, maxval=255, type=cv2.THRESH_BINARY)[1]
# cv2.imshow('mask', mask)

# maska odwrotna białe ROI czarne tło
mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask_inv', mask_inv)
# cv2.waitKey(0)

# wyciecie tla obrazu
img_bg = cv2.bitwise_and(src1=roi, src2=roi, mask=mask)
cv2.imshow('img_bg', img_bg)
# wyciecie loga
logo_fg = cv2.bitwise_and(src1=logo, src2=logo, mask=mask_inv)
cv2.imshow('logo_fg', logo_fg)

# dodanie obrazow
dst = cv2.add(src1=img_bg, src2=logo_fg)
cv2.imshow('dst', dst)
img[:rows, :cols] = dst
cv2.imshow('out', img)
cv2.waitKey(0)