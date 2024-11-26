package cn.zbq.onnx_demo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.HashMap;
import java.util.Map;
import java.util.ResourceBundle;

/**
 * @author zbq
 * @since 2024/11/4
 */
public class OnnxTest {

    public static void main(String[] args) {
        String modelPath = ResourceBundle.getBundle("application").getString("app.data") + "/model/demo.onnx";

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions();
             OrtSession session = env.createSession(modelPath, options)) {
            // options.addCUDA(); // cuda
            float[][] inputData = new float[][]{
                    {-0.8434235f, 0.38335258f, 0.4419144f, 0.8077471f, 1.5492866f, -0.07422484f, 0.3821767f, -0.36659962f,
                            -0.21357884f, -1.004654f, 0.0070992415f, -0.19587615f, -0.67492676f, 0.95546705f, 0.89641213f, 0.8950324f}
            };

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            // 动态获取，根据实际情况确定,这里只取第一个
            String first = session.getInputNames().iterator().next();
            inputs.put(first, inputTensor);
            OrtSession.Result result = session.run(inputs);
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            float[][] outputData = (float[][]) outputTensor.getValue();

            // 打印输出
            System.out.println("Model Output:");
            for (float[] outputs : outputData) {
                for (float value : outputs) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }

            // 释放资源
            outputTensor.close();
            inputTensor.close();

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
}
