package com.example.fufamilydeskwin7.svm;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "OCVSample::Activity";
    private Scalar tains[];
    private float [][]tain ={{10,10},{35,2},{50,50},{60,10},{20,60},
            {100,500},{70,50},{80,200},{60,90},{10,300}};
    private double []labels = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,1.0};
    private Boolean svmBoo;
    private Mat trainingDataMat;
    private Mat responsesMat;
    private CvSVMParams params;
    private CvSVM svm;

    private TextView information;
    private EditText inputx, inputy;
    private Button check;

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
        /*params = new CvSVMParams();
        Log.i(TAG, "new CvSVMParams");
        params.set_svm_type(CvSVM.C_SVC);
        Log.i(TAG, "svm_type");
        params.set_kernel_type(CvSVM.LINEAR);
        Log.i(TAG, "kernel_type");
        params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 100, 1e-6));
        Log.i(TAG, "params finish");*/
        svm = new CvSVM(trainingDataMat, responsesMat);
        Log.i(TAG, "new CvSVM");
//        svmBoo=svm.train(trainingDataMat, responsesMat, new Mat(), new Mat(), params);
//        svmBoo = svm.train(trainingDataMat, responsesMat);
        Log.i(TAG, "SVM");
        Log.i(TAG, "SVM123");
//        StringBuilder text = new StringBuilder();
//        double[] data;
//        for (int row = 0; row <= 8; row++) {
//            for (int col = 0; col <= 1; col++)
//                data = trainingDataMat.get(row, col);
////                text.append(String.valueOf(data[0][0]));
//        }
//        information.setText(text);

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

    }

}
