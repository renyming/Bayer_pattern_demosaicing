#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printMat(const Mat& mat);
void showComparison(const String& title, const Mat& orig, const Mat& result);
void printHelp(const String&);
void savePatchComparison(const String&, const Mat&, const Mat&, const Mat&);

int main(int argc, char* argv[]) {

	if (argc < 3) {
		printHelp(argv[0]);
		return -1;
	}

	Mat img, original;
	img = imread(argv[1], CV_8U);
	original = imread(argv[2]);

	if (!img.data || !original.data) {
		printHelp(argv[0]);
		return -1;
	}

	int cols = img.cols;
	int rows = img.rows;

	Mat bayerBlue = Mat::zeros(rows, cols, CV_32F);
	Mat bayerRed = Mat::zeros(rows, cols, CV_32F);
	Mat bayerGreen = Mat::zeros(rows, cols, CV_32F);

	bool isOddRow = true;
	for (int i = 0; i < rows; ++i) {
		bool isOddCol = true;
		for (int j = 0; j < cols; ++j) {
			unsigned char bayerValue = img.at<uchar>(i, j);
			if (isOddRow) {
				if (isOddCol) {
					bayerBlue.at<float>(i, j) = bayerValue;
				}
				else {
					bayerRed.at<float>(i, j) = bayerValue;
				}
			}
			else {
				if (isOddCol) {
					bayerRed.at<float>(i, j) = bayerValue;
				}
				else {
					bayerGreen.at<float>(i, j) = bayerValue;
				}
			}
			isOddCol = !isOddCol;
		}
		isOddRow = !isOddRow;
	}


	float blueKernelData[9] = { 1 / 4.0f,1 / 2.0f,1 / 4.0f,1 / 2.0f,1.0f,1 / 2.0f,1 / 4.0f,1 / 2.0f,1 / 4.0f };
	Mat blueKernel = Mat(3, 3, CV_32F, blueKernelData);
	Mat blue;
	filter2D(bayerBlue, blue, -1, blueKernel);

	float redKernelData[9] = { 0.0f,1 / 4.0f,0.0f,1 / 4.0f,1.0f,1 / 4.0f,0.0f,1 / 4.0f,0.0f };
	Mat redKernel = Mat(3, 3, CV_32F, redKernelData);
	Mat red;
	filter2D(bayerRed, red, -1, redKernel);

	float greenKernelData[9] = { 1 / 4.0f,1 / 2.0f,1 / 4.0f,1 / 2.0f,1.0f,1 / 2.0f,1 / 4.0f,1 / 2.0f,1 / 4.0f };
	Mat greenKernel = Mat(3, 3, CV_32F, greenKernelData);
	Mat green;
	filter2D(bayerGreen, green, -1, greenKernel);

	vector<Mat> bgrImage;
	bgrImage.push_back(blue);
	bgrImage.push_back(green);
	bgrImage.push_back(red);

	Mat image;
	merge(bgrImage, image);
	image.convertTo(image, CV_8UC3);

	showComparison("Part 1", original, image);

	/*Part 2*/
	Mat diffG_R = Mat(rows, cols, CV_32F);
	diffG_R = green - red;
	Mat diffB_R = Mat(rows,cols,CV_32F);
	diffB_R = blue - red;

	Mat G_R= Mat(rows, cols, CV_32F);
	medianBlur(diffG_R, G_R, 3);

	Mat B_R= Mat(rows, cols, CV_32F);
	medianBlur(diffB_R, B_R, 3);

	G_R = G_R + red;
	B_R = B_R + red;

	bgrImage.clear();
	bgrImage.push_back(B_R);
	bgrImage.push_back(G_R);
	bgrImage.push_back(red);

	Mat imgImproved;
	merge(bgrImage, imgImproved);
	imgImproved.convertTo(imgImproved, CV_8UC3);

	showComparison("Part 2", original, imgImproved);
	
	return 0;
}

void showComparison(const String& title, const Mat& orig, const Mat& result) {

	int rows = orig.rows;
	int cols = orig.cols;

	Mat diff = Mat(rows, cols, CV_8UC3);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			Vec3b oPixel = orig.at<Vec3b>(i, j);
			Vec3b rPixel = result.at<Vec3b>(i, j);
			diff.at<Vec3b>(i, j) = Vec3b(
				static_cast<uchar>(sqrt(abs(pow(oPixel[0], 2) - pow(rPixel[0], 2)))), 
				static_cast<uchar>(sqrt(abs(pow(oPixel[1], 2) - pow(rPixel[1], 2)))),
				static_cast<uchar>(sqrt(abs(pow(oPixel[2], 2) - pow(rPixel[2], 2))))
			);
		}
	}

	Mat comparison;
	hconcat(orig, result, comparison);
	hconcat(comparison, diff, comparison);
	
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, comparison);

	//savePatchComparison(title+" patch(2x scale)", orig, result, diff);
	//cout << "Difference sum of " << title << ": " << sum(diff) << endl;

	waitKey();

}

//function to save the close-up patch comparison image
void savePatchComparison(const String& title, const Mat& orig, const Mat& result, const Mat& diff) {
	Rect patch(215, 490, 80, 55);
	Mat cropOriginal = orig(patch);
	Mat cropResult = result(patch);
	Mat cropDiff = diff(patch);

	resize(cropOriginal, cropOriginal, Size(160, 110));
	resize(cropResult, cropResult, Size(160, 110));
	resize(cropDiff, cropDiff, Size(160, 110));

	Mat patchComparison;
	hconcat(cropOriginal, cropResult, patchComparison);
	hconcat(patchComparison, cropDiff, patchComparison);
	imwrite(title + ".bmp", patchComparison);
}

void printMat(const Mat& mat) {
	int rows = mat.rows;
	int cols = mat.cols;

	cout << setw(3);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cout << setw(3) << (int)mat.at<uchar>(i, j) << setw(1) << ' ';
		}
		cout << endl;
	}

}

void printHelp(const String& programName) {
	cout << "Please specify a Bayer raw image file and an original file" << endl;
	cout << "Usage: " << programName << ' ' << "[raw file name]" << ' ' << "[original file name]" << endl;
}