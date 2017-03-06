#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// function based upon http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html

float getMSSIM(const Mat& i1, const Mat& i2, const Mat& importance_map)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    float sum = 0;
    float count = 0;

    for(int i=0; i < i1.rows; i++)
    {
        for(int j=0; j < i1.cols; j++)
        {
            if (importance_map.at<uchar>(i,j) > 0)
            {
                sum += ssim_map.at<float>(i,j);
                count += 1;
            }
        }
    }

    float mssim = sum / count;

    // Scalar mssim = mean(ssim_map);   // mssim = average of ssim map

    return mssim;
}

int main(int argc, char** argv )
{
    int scale = 1;
    int delta = 0;

    if ( argc != 3 )
    {
        printf("usage: DisplayImage.out <Image_Path> <Image_Path>\n");
        return -1;
    }

    Mat img_src;
    Mat img_compressed;

    img_src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    img_compressed = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    img_src.convertTo(img_src, CV_64F);
    img_compressed.convertTo(img_compressed, CV_64F);

    if ( !img_src.data || !img_compressed.data )
    {
        printf("Problem with input image data, exiting. \n");
        return -1;
    }

    // sobel example: http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel(img_src, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel(img_src, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs( grad_y, abs_grad_y );

    Mat grad;

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    float mssim = getMSSIM(img_src, img_compressed, grad);

    printf("%f\n", mssim);

    return 0;
}
