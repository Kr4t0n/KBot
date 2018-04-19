package talk;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import main.TalkFrame;

public class Initialize {
	
	
	static {
		File fp=new File("config");
		FileReader fr;
		try {
			fr = new FileReader(fp);
			BufferedReader br=new BufferedReader(fr);
			PY_DIR=br.readLine();
			BOT_DIR=br.readLine();
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
	}
	
	/**
	 * 只需配置以下三个路径
	 */
	private static String PY_NAME="chatbot.pyc";//py文件名
	private static String PY_DIR;//python可执行文件的 绝对路径
	private static String BOT_DIR;//chatbot 的绝对路径
	private static TalkFrame tf;
	
	public Initialize(TalkFrame tf) {
		this.tf=tf;
		start();
	}
	
	public void start() {
		Process pr;
		try {
			pr = Runtime.getRuntime().exec(PY_DIR+" "+PY_NAME+" --mode chat",null,new File(BOT_DIR));
			 BufferedReader in = new BufferedReader(new  InputStreamReader(pr.getInputStream()));  
		        String line; 
		        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(pr.getOutputStream())), true);
		        MyTalk.pr=pr;
		        MyTalk.out=out;
		        MyTalk.in=in;
		        while ((line = in.readLine()) != null) {  
		            System.out.println(line); 
		            if(line.contains("Welcome, COMP7404! I'm a chatbot!")) {
		            		tf.enableGUI();
		            		break;
		            }
		            
		        }  
		        //in.close();  
		        pr.waitFor();
		        //System.out.println(pr.waitFor()); 
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		
        
       
	}

}
