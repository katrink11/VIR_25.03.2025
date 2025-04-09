#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
	// Загрузка изображений
	string image_path1 = "./part1.png";
	string image_path2 = "./part2.png";

	Mat img1 = imread(image_path1, IMREAD_COLOR);
	Mat img2 = imread(image_path2, IMREAD_COLOR);

	if (img1.empty() || img2.empty())
	{
		cerr << "Ошибка: не удалось загрузить изображения!" << endl;
		return -1;
	}

	// Сохраняем оригинальные изображения (для отладки)
	imwrite("01_original1.jpg", img1);
	imwrite("02_original2.jpg", img2);

	// Инициализация ORB детектора
	Ptr<ORB> orb = ORB::create(5000);
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Поиск ключевых точек и вычисление дескрипторов
	orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	// Визуализация ключевых точек
	Mat img_keypoints1, img_keypoints2;
	drawKeypoints(img1, keypoints1, img_keypoints1);
	drawKeypoints(img2, keypoints2, img_keypoints2);
	imwrite("03_keypoints1.jpg", img_keypoints1);
	imwrite("04_keypoints2.jpg", img_keypoints2);

	// Сопоставление дескрипторов
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> knn_matches;
	matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

	// Фильтрация совпадений
	vector<DMatch> good_matches;
	const float ratio_thresh = 0.7f;
	for (const auto &match : knn_matches)
	{
		if (match.size() == 2 && match[0].distance < ratio_thresh * match[1].distance)
		{
			good_matches.push_back(match[0]);
		}
	}

	// Визуализация всех совпадений
	Mat img_all_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_all_matches);
	imwrite("05_all_matches.jpg", img_all_matches);

	// Преобразование ключевых точек
	vector<Point2f> points1, points2;
	for (const auto &match : good_matches)
	{
		points1.push_back(keypoints1[match.queryIdx].pt);
		points2.push_back(keypoints2[match.trainIdx].pt);
	}

	// Вычисление гомографии
	Mat H = findHomography(points2, points1, RANSAC);

	// Визуализация гомографии
	Mat img_homography;
	vector<Point2f> img2_corners(4), img2_transformed_corners(4);
	img2_corners[0] = Point2f(0, 0);
	img2_corners[1] = Point2f(img2.cols, 0);
	img2_corners[2] = Point2f(img2.cols, img2.rows);
	img2_corners[3] = Point2f(0, img2.rows);
	perspectiveTransform(img2_corners, img2_transformed_corners, H);

	img1.copyTo(img_homography);
	for (int i = 0; i < 4; i++)
	{
		line(img_homography, img2_transformed_corners[i], img2_transformed_corners[(i + 1) % 4], Scalar(0, 255, 0), 4);
	}
	imwrite("06_homography.jpg", img_homography);

	// Создание панорамы
	Mat panorama;
	warpPerspective(img2, panorama, H, Size(img1.cols + img2.cols, img1.rows));
	imwrite("07_warped_image.jpg", panorama); // Промежуточный результат

	img1.copyTo(panorama(Rect(0, 0, img1.cols, img1.rows)));
	imwrite("08_combined_panorama.jpg", panorama); // Перед обрезкой

	// Обрезка черных областей
	Rect crop_rect(0, 0, panorama.cols, panorama.rows);
	for (int i = 0; i < panorama.cols; i++)
	{
		if (panorama.at<Vec3b>(panorama.rows / 2, i) != Vec3b(0, 0, 0))
		{
			crop_rect.x = i;
			break;
		}
	}
	for (int i = panorama.cols - 1; i >= 0; i--)
	{
		if (panorama.at<Vec3b>(panorama.rows / 2, i) != Vec3b(0, 0, 0))
		{
			crop_rect.width = i - crop_rect.x;
			break;
		}
	}

	Mat cropped_panorama = panorama(crop_rect);

	// Сохранение и отображение результатов
	if (!imwrite("09_final_panorama.jpg", cropped_panorama))
	{
		cerr << "Ошибка при сохранении финальной панорамы!" << endl;
		return -1;
	}

	cout << "Все промежуточные результаты сохранены в файлы:" << endl;
	cout << "01-09_*.jpg" << endl;

	imshow("Final Panorama", cropped_panorama);
	waitKey(0);

	return 0;
}