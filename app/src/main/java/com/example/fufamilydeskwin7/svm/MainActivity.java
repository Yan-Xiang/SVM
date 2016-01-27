package com.example.fufamilydeskwin7.svm;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "OCVSample::Activity";
    private Scalar tains[];
//    private float[][] tain ={{10,10},{35,2},{50,50},{60,10},{20,60},
//            {120,150},{150,140},{120,60},{100,120},{90,100}};
//    private float[][]tain={{10,10},{2,35},{50,50},{10,60},{60,20},
//            {150,120},{140,150},{60,120},{120,100},{100,90}};
    private float[][]tain={{10,10},{2,35},{50,50},{10,60},{60,20},
            {20,120},{40,150},{60,120},{20,100},{10,90}};
    private float []labels = {-1, -1, -1, -1, -1,
            1, 1, 1, 1, 1};
    private Boolean svmBoo;
    private Mat trainingDataMat;
    private Mat responsesMat;
    private CvSVMParams params;
    private CvSVM svm;


    private TextView information;
    private EditText inputx, inputy;
    private Button check;
    private Bitmap showbm;
    private ImageView showimage;

//    private static final String
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        layout_set();
//        tains = new Scalar[]{ new Scalar(10, 10), new Scalar(35, 2), new Scalar(50, 50), new Scalar(60, 10), new Scalar(20, 60), new Scalar(100, 500), new Scalar(70, 50), new Scalar(80, 200), new Scalar(60, 90) };

//        Scalar tainingData = new Scalar(tain);


    }

    public void buttonClick(View v) {
        Log.i(TAG, "onClick");
        trainingDataMat = new Mat(10, 2, CvType.CV_32FC1);
        Log.i(TAG, "new Mat");
        for (int row = 0; row <= 9; row++) {
            Log.i(TAG, "for loop row tain to Mat");
            for (int col = 0; col <= 1; col++) {
                Log.i(TAG, "for loop col tain to Mat");
                trainingDataMat.put(row, col, tain[row][col]);

            }
        }
        Log.i(TAG, "tain to Mat finish");
        responsesMat = new Mat(10, 1, CvType.CV_32FC1);
        Log.i(TAG, "new Mat-responsesMat");
        for (int row = 0; row <= 9; row++) {
            Log.i(TAG, "for loop row labels to Mat");
            for (int col = 0; col < 1; col++) {
                Log.i(TAG, "for loop col labels to Mat");
                responsesMat.put(row,col, labels[row]);
            }
        }
        Log.i(TAG, "labels to Mat finish");
//        svm = new CvSVM(trainingDataMat, responsesMat);
        svm = new CvSVM();

        params = new CvSVMParams();
        Log.i(TAG, "new CvSVMParams");
        //設定CvSVMParams
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_term_crit(new TermCriteria(TermCriteria.EPS|TermCriteria.MAX_ITER, 100, 1e-6));
        Log.i(TAG, "params finish");

        Log.i(TAG, "new CvSVM");
//        svmBoo=svm.train(trainingDataMat, responsesMat, new Mat(), new Mat(), params);
        //做 SVM 訓練
        svm.train_auto(trainingDataMat, responsesMat, new Mat(), new Mat(), params);

//        svmBoo = svm.train(trainingDataMat, responsesMat);
        Log.i(TAG, "SVM train OK");

//        StringBuilder text = new StringBuilder();
//        double[] data;
//        for (int row = 0; row <= 8; row++) {
//            for (int col = 0; col <= 1; col++)
//                data = trainingDataMat.get(row, col);
////                text.append(String.valueOf(data[0][0]));
//        }
//        information.setText(text);

        //標記訓練資料點的位置
        Mat show32 = new Mat(160,160,CvType.CV_32SC3);
        Size show32size = show32.size();
        double[] green = {100, 255, 100};
        double[] red = {255, 70, 70};

        for (int row = 0; row <= show32size.height; row++) {
            for (int col = 0; col <= show32size.width; col++) {
//                Log.i(TAG, "response "+String.valueOf(row)+" , "+String.valueOf(col));
                //輸入圖片各個點的座標
                Mat input=new Mat(1,2,CvType.CV_32FC1, new Scalar(row,col));
                //一一比對，讓每個點都判斷過，畫出整張圖的結果
                float response = svm.predict(input);


                if (response == 1) {
                    show32.put(row,col,green );
                } else if (response == -1) {
                    show32.put(row, col, red);
                }
//                Log.i(TAG, "response value: " + String.valueOf(response));
            }
        }
        double[] whiet = {255, 255, 255};
        double[] black = {0, 0, 0};
        for (int tainrow = 0; tainrow <= 9; tainrow++) {
//            for (int taincol = 0; taincol <= 1; taincol++) {
                for (int row = 0; row < 5; row++) {
                    for (int col = 0; col < 5; col++) {
                        if (labels[tainrow] == 1) {
                            int putrow = (int)tain[tainrow][0] + row - 2;
                            show32.put(putrow, (int)tain[tainrow][1] + col - 2, whiet);

                        }
                        else if (labels[tainrow] == -1) {
                            show32.put((int)tain[tainrow][0] + row - 2, (int)tain[tainrow][1] + col - 2, black);
                        }
                    }
                }
//            }
        }


        Log.i(TAG, "response OK");
        Mat show = new Mat(show32size, CvType.CV_8UC3);
        Log.i(TAG, "new Mat : show");
        show32.convertTo(show, CvType.CV_8UC3);
        Imgproc.resize(show, show, new Size(600, 600));
        Log.i(TAG, "convertTo OK");
        Size showsize = show.size();
        showbm = Bitmap.createBitmap((int)showsize.width,(int)showsize.height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(show, showbm);
        Log.i(TAG, "mattobitmap");
        showimage.setImageBitmap(showbm);


    }

    public void checkClick(View v) {
//        inputx
//                inputy
    }
    //OpenCV类库加载并初始化成功后的回调函数，在此我们不进行任何操作
        private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                Log.i(TAG, "onManagerConnected star ");
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: {
                        Log.i(TAG, "OpenCV loaded successfully");

                    }
                    break;
                    default: {
                        super.onManagerConnected(status);
                    }
                    break;
                }
            }
        };
    /** Call on every application resume **/
    @Override
    protected void onResume()
    {
        Log.i(TAG, "Called onResume");
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, this, mLoaderCallback);

        Log.i(TAG, " onResume OK");
    }
    public void layout_set() {
        information=(TextView) findViewById(R.id.Text);
        inputx = (EditText) findViewById(R.id.inputx);
        inputy = (EditText) findViewById(R.id.inputy);
        showimage = (ImageView) findViewById(R.id.showimage);
    }

}
