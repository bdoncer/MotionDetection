import cv2
import numpy


def display_frames(frame1, frame2, frame3):
    cv2.imshow("Gray Frame", frame1)
    cv2.imshow("Difference Frame", frame2)
    cv2.imshow("Threshold Frame", frame3)


if __name__ == '__main__':
    is_debug = input("Czy chcesz włączyć tryb debug? (y lub n): ")
    name = input("Ustal źródło (0 - kamera): ")
    if name == '0':
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(name)
    sens = int(input("Wprowadź wartość czułości: (0-255) "))
    upgrade = input("Czy chcesz określić obszar czułości? (y lub n): ")
    if upgrade == 'y':
        masks_nr = int(input("Ile obszarów czułości chcesz określić?: "))
        start_x = []
        start_y = []
        pref_width = []
        pref_height = []
        for i in range(int(masks_nr)):
            print("Obszar czułości nr ",i+1)
            start_point = input("Punkt startowy dla obszaru wychwytywania (np. 20 20): ")
            pref_w = int(input("Szerokość obszaru wychwytywania ruchu (max 640): "))
            pref_h = int(input("Wysokość obszaru wychwytywania ruchu (max 480): "))
            start_point = start_point.split()
            start_x.append(int(start_point[0]))
            start_y.append(int(start_point[1]))
            pref_height.append(pref_h)
            pref_width.append(pref_w)

    else:
        masks_nr = 1
        start_x = [0]
        start_y = [0]
        pref_width = [-1]
        pref_height = [-1]

    prev_frame = None
    cnt = 0


    while 1:
        frame = video.read()[1]
        full_width = len(frame[0])
        full_height = len(frame)
        if pref_height == -1:
            pref_width = len(frame[0])
            pref_height = len(frame)

        contour_width = []
        contour_height = []
        for i in range(masks_nr):
            contour_width.append([start_x[i], start_x[i] + pref_width[i]])
            contour_height.append([start_y[i], start_y[i] + pref_height[i]])
            cv2.rectangle(frame, (int(contour_width[i][0]), int(contour_height[i][0])),
                          (int(contour_width[i][1]), int(contour_height[i][1])), (0, 255, 0), 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if prev_frame is not None:
            for i in range(masks_nr):
                cv2.accumulateWeighted(gray_frame, prev_frame, 0.05)
                diff_frame = cv2.absdiff(
                    cv2.convertScaleAbs(prev_frame)[start_y[i]:start_y[i] + pref_height[i], start_x[i]:start_x[i] + pref_width[i]],
                    gray_frame[start_y[i]:start_y[i] + pref_height[i], start_x[i]:start_x[i] + pref_width[i]])
                thresh_frame = cv2.threshold(diff_frame, sens, 255, cv2.THRESH_BINARY)[1]
                thresh_frame = cv2.dilate(thresh_frame, None, iterations=1)
                contour_frame = thresh_frame.copy()
                diff_frame_debug = cv2.absdiff(cv2.convertScaleAbs(prev_frame), gray_frame)
                thresh_frame_debug = cv2.threshold(diff_frame_debug, sens, 255, cv2.THRESH_BINARY)[1]
                moves = cv2.findContours(contour_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                for move in moves:
                    (x, y, w, h) = cv2.boundingRect(move)
                    if cv2.contourArea(move) > 1000:
                        cv2.rectangle(frame, (int(contour_width[i][0] + x), int(contour_height[i][0] + y)),
                                      (int(contour_width[i][0] + x + w), int(contour_height[i][0] + y + h)), (128, 0, 200), 2)


            if is_debug == "y":
                display_frames(gray_frame, diff_frame_debug, thresh_frame_debug)
            cv2.imshow("Color Frame", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            prev_frame = gray_frame.copy().astype("float")

    video.release()
    cv2.destroyAllWindows()



