package com.bmh.minhhieu.opencvandroid;

import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.Text;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.HttpCookie;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "test";
    private static final String TAG1 = "test1";
    Button btn;
    ImageView imgView;
    Button chooseImg;
    TextView textView;
    ImageView imgView2;

    BitmapDrawable drawable1;
    BitmapDrawable drawable2;
    Bitmap bitmap1;
    Bitmap bitmap2;
    String imgString1="";
    String imgString2="";
    Drawable myIcon;
    ArrayList arrayList;
    ArrayList labelList;
    ArrayList lhdList;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn = (Button)findViewById(R.id.btn);
        chooseImg = (Button) findViewById(R.id.chooseImg);
        imgView = (ImageView)findViewById(R.id.imgView);
        textView = (TextView)findViewById(R.id.textView);
        imgView2 = (ImageView)findViewById(R.id.imgView2);
        arrayList = new ArrayList<>();
        labelList = new ArrayList<>();
        lhdList = new ArrayList<>();

        DataArray();



//        btn.setOnClickListener(onClickListener);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mat src = new Mat();
                Mat dst = new Mat();
                Mat grayImage = new Mat();
                try {
                    src = Utils.loadResource(MainActivity.this, R.drawable.minion, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                    Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_RGB2GRAY);
                    Imgproc.blur(grayImage, grayImage, new Size(3, 3));
                    Imgproc.Canny(grayImage, dst, 20, 60, 3, false);
//                    Imgproc.HoughLinesP(grayImage, dst, 180, Math.PI / 180, 50, 0, 0);

                    Bitmap img = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(dst, img);
                    imgView.setImageBitmap(img);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform(this));
        }

//        chooseImg.setOnClickListener(onClickListener);
        final Python py = Python.getInstance();

        chooseImg.setOnClickListener((new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                drawable1 = (BitmapDrawable)imgView.getDrawable();
//                bitmap1 = drawable1.getBitmap();
                drawable2 = (BitmapDrawable)imgView2.getDrawable();
                bitmap2 = drawable2.getBitmap();
                imgString2 = getStringImage(bitmap2);
//                myIcon = getResources().getDrawable(R.drawable.minion);
                for (int i=1; i<=3;i++){
                    bitmap1 = (Bitmap) arrayList.get(i);
                    imgString1 = getStringImage(bitmap1);
                    // now in imageString we get encoded image string
                    // now i will pass this string in python script
                    PyObject pyobj = py.getModule("script1");
                    // so here i call main method of script and pass image string as parameter..
                    PyObject obj = pyobj.callAttr("main1",imgString1, imgString2);
                    // so obj will contain return value ...i.e our image string..

                    labelList.add(obj);
                }
                textView.setText(labelList.toString());
//                String str = obj.toString();
//                byte data[] = android.util.Base64.decode(str, Base64.DEFAULT);
//                // now convert it to bitmap
//                Bitmap bmp = BitmapFactory.decodeByteArray(data, 0, data.length);
//                imgView2.setImageBitmap(bmp);
            }
        }));
    }
    public void DataArray(){
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.buiminhhieu_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.chipu_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.dinhhoanghieu_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.duongminhnhat_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.doanhuutoan_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.letutuan_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.luuphanhiep_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.maianhvu_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.maiphuongthuy_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.ribi_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.nguyenhoangan_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.nguyenngocdien_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.nguyenquanghieu_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.nguyenquocdat_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.phanhoangnguyen_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.minhhang_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.phantienminh_0)));
        arrayList.add((BitmapFactory.decodeResource(getResources(), R.drawable.tuyetchinh_0)));
    }


    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);

        byte[] imageBytes = baos.toByteArray();
        // finally encode to String
        String encodeImage = android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return  encodeImage;
    }


    static {
        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV loaded");
        }
        else {Log.d(TAG, "OpenCV not loaded");}
    }

    }
