// featuresCal.cpp
//
// Show the usage of various keypoints detectors, extractors and matchers.
//
// Calling convention:
// ./featuresCal imgA.JPG imgB.JPG

/* ******************************************************************* */
/* ------------------------------------------------------------------- */
/* Pure detector */
/* ------------------------------------------------------------------- */
// Harris-Shi-Tomasi corner detector
// Blob detector
// FAST detector
// Star/CenSrE detector
/* ------------------------------------------------------------------- */
/* Pure extractor */
/* ------------------------------------------------------------------- */
// BRIEF extractor
// FREAK extractor
/* ------------------------------------------------------------------- */
/* Both detector and extractor */
/* ------------------------------------------------------------------- */
// SIFT detector
// SURF detector and extractor
// BRISK detector and extractor
// ORB detector and extractor
/* ------------------------------------------------------------------- */
/* Pure matcher */
/* ------------------------------------------------------------------- */
// BFMatcher
// FLANN matcher
/* ******************************************************************* */


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int main (int argc, char** argv){

    // Initialize, load two images from the file system, and
    // allocate the images and other structures we will need for
    // results.
    //
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    //
    cv::Size img_sz = img_1.size();
    cv::Mat outImg_1, outImg_2, outImg;
    //
    // Detector definition.
    cv::Ptr<cv::Feature2D> f2d;
    // Step 1: Detect the keypoints.
    vector< cv::KeyPoint > keypoints_1, keypoints_2;
    // Step 2: Extract descriptors (feature vectors).
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    // Step 3: Matching descriptor vectors using BFMatcher.
    cv::BFMatcher matcher_bf;
    cv::FlannBasedMatcher matcher_flann;
    vector < cv::DMatch > matches;


    /* ******************************************************************* */
    /* ******************************************************************* */
    /* Pure detector */
    /* ******************************************************************* */
    /* ******************************************************************* */

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Harris-Shi-Tomasi corner detector + ORB extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::GFTTDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::ORB::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);
    
    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Harris_Shi_Tomasi-Features-ORB-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("Harris_Shi_Tomasi-Features-ORB-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring Harris_Shi_Tomasi-Features-ORB-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-Harris-ORB-BF.JPG", outImg_1);
    cv::imwrite("results/matches-Harris-ORB-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // // cv::destroyAllWindows();

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Blob detector  + ORB extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::SimpleBlobDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::ORB::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Blob-Features-ORB-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("Blob-Features-ORB-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring Blob-Features-ORB-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-Blob-ORB-BF.JPG", outImg_1);
    cv::imwrite("results/matches-Blob-ORB-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();    

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // FAST detector  + ORB extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::FastFeatureDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::ORB::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("FAST-Features-ORB-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("FAST-Features-ORB-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring FAST-Features-ORB-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-FAST-ORB-BF.JPG", outImg_1);
    cv::imwrite("results/matches-FAST-ORB-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();    

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Star/CenSrE detector + ORB extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::xfeatures2d::StarDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::ORB::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("StarDetector-Features-ORB-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("StarDetector-Features-ORB-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring StarDetector-Features-ORB-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-StarDetector-ORB-BF.JPG", outImg_1);
    cv::imwrite("results/matches-StarDetector-ORB-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();     


    /* ******************************************************************* */
    /* ******************************************************************* */
    /* Pure descriptor extractor*/
    /* ******************************************************************* */
    /* ******************************************************************* */

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Harris-Shi-Tomasi corner detector + BRIEF extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::GFTTDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Harris_Shi_Tomasi-Features-BRIEF-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("Harris_Shi_Tomasi-Features-BRIEF-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring Harris_Shi_Tomasi-Features-BRIEF-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-Harris-BRIEF-BF.JPG", outImg_1);
    cv::imwrite("results/matches-Harris-BRIEF-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();    

    
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Harris-Shi-Tomasi corner detector + FREAK extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::GFTTDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::xfeatures2d::FREAK::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Harris_Shi_Tomasi-Features-FREAK-Extractor-BF-Matcher", outImg_1);
    cv::moveWindow("Harris_Shi_Tomasi-Features-FREAK-Extractor-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring Harris_Shi_Tomasi-Features-FREAK-Extractor-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-Harris-FREAK-BF.JPG", outImg_1);
    cv::imwrite("results/matches-Harris-FREAK-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;    
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();     



    /* ******************************************************************* */
    /* ******************************************************************* */
    /* Both detector and extractor*/
    /* ******************************************************************* */
    /* ******************************************************************* */

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // SIFT detector and extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::xfeatures2d::SIFT::create();

    // Step 1 and 2: Detect the keypoints & Extract descriptors.
    f2d->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    f2d->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("SIFT-Features-BF-Matcher", outImg_1);
    cv::moveWindow("SIFT-Features-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring SIFT-Features-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-SIFT-BF.JPG", outImg_1);
    cv::imwrite("results/matches-SIFT-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();     


    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // SURF detector and extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::xfeatures2d::SURF::create();

    // Step 1 and 2: Detect the keypoints & Extract descriptors.
    f2d->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    f2d->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("SURF-Features-BF-Matcher", outImg_1);
    cv::moveWindow("SURF-Features-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring SURF-Features-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-SURF-BF.JPG", outImg_1);
    cv::imwrite("results/matches-SURF-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;  
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();       



    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // BRISK detector and extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::BRISK::create();

    // Step 1 and 2: Detect the keypoints & Extract descriptors.
    f2d->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    f2d->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("BRISK-Features-BF-Matcher", outImg_1);
    cv::moveWindow("BRISK-Features-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring BRISK-Features-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-BRISK-BF.JPG", outImg_1);
    cv::imwrite("results/matches-BRISK-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;   
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();         

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // ORB detector and extractor + BFMatcher
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::ORB::create();

    // Step 1 and 2: Detect the keypoints & Extract descriptors.
    f2d->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
    f2d->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 3: Matching descriptor vectors using BFMatcher.
    matcher_bf.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("ORB-Features-BF-Matcher", outImg_1);
    cv::moveWindow("ORB-Features-BF-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring ORB-Features-BF-Matcher..." << endl;
    cv::imwrite("results/keypoints-ORB-BF.JPG", outImg_1);
    cv::imwrite("results/matches-ORB-BF.JPG", outImg);
    cout << "\n---Done!---\n" << endl;  
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();     

    /* ******************************************************************* */
    /* ******************************************************************* */
    /* Pure matcher */
    /* ******************************************************************* */
    /* ******************************************************************* */
    
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // Harris-Shi-Tomasi corner detector + FREAK extractor + FLANN
    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    f2d = cv::GFTTDetector::create();

    // Step 1: Detect the keypoints.
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);

    // Display results.
    cv::drawKeypoints(
        img_1, 
        keypoints_1, 
        outImg_1, 
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Step 2: Extract descriptors (feature vectors).
    extractor = cv::xfeatures2d::FREAK::create();
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    // Step 3: Matching descriptor vectors using BFMatcher.
    if(descriptors_1.type()!=CV_32F) {
        descriptors_1.convertTo(descriptors_1, CV_32F);
    }
    if(descriptors_2.type()!=CV_32F) {
        descriptors_2.convertTo(descriptors_2, CV_32F);
    }
    matcher_flann.match(descriptors_1, descriptors_2, matches);

    // Display results.
    cv::drawMatches(
        img_1, 
        keypoints_1, 
        img_2, 
        keypoints_2, 
        matches, 
        outImg,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cv::imshow("Harris_Shi_Tomasi-Features-FREAK-Extractor-FALNN-Matcher", outImg_1);
    cv::moveWindow("Harris_Shi_Tomasi-Features-FREAK-Extractor-FALNN-Matcher", 0, 0);
    // cv::imshow("outImg", outImg);

    cout << "\nStoring Harris_Shi_Tomasi-Features-FREAK-Extractor-FLANN-Matcher..." << endl;
    cv::imwrite("results/keypoints-Harris-FREAK-FLANN.JPG", outImg_1);
    cv::imwrite("results/matches-Harris-FREAK-FLANN.JPG", outImg);
    cout << "\n---Done!---\n" << endl;
    if(cv::waitKey(1000) == 27) exit(-1);
    // cv::destroyAllWindows();     


    
    cv::destroyAllWindows();
    return 0;

}