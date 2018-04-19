package talk;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

import entities.IMMessage;
import main.TalkFrame;


public class MyTalk {
	
	
	private StringBuilder message=new StringBuilder(30);
	
	public static Process pr;
	public static PrintWriter out;
	public static BufferedReader in;
	public static TalkFrame tf;
	
	public IMMessage say(String a) {
		
		out.println(a);
		String line;
		StringBuilder context;
		IMMessage imMsg = new IMMessage();
		int i=1;
		try {
			while ((line = in.readLine()) != null) {  
			    System.out.println(i+line);
			    i++;
			    if(!(line.contains("[")||line.contains("]"))) {
			    		context=new StringBuilder(line.length());
			    		int index=0;
			    		for(char c:line.toCharArray()) {
			    			if(index==0) {
			    				if(c!=','&&c!='>'&&c!=' ') {
			    					context.append(Character.toUpperCase(c));
			    				}
			    				else {
			    					continue;
			    				}
			    				
			    			}else {
			    				context.append(c);
			    			}
			    			index++;
			    		}
			    		String sSend = "Chatbot";
			    		String sTime =  getTime();
			    		
			    		imMsg.setSender(sSend);
			    		imMsg.setTime(sTime);
			    		imMsg.setMsg(context.toString());
			    		return imMsg;
			 

			    }
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return imMsg;
            
        
	}
	

	
	
	static String getTime() {
		SimpleDateFormat sdf = new SimpleDateFormat("hh:mm:ss");
		return sdf.format(new Date());
	}
	public IMMessage getMessage() {
		
		
		
		String sSend = "Chatbot";
		String sTime =  getTime();
		IMMessage imMsg = new IMMessage();
		imMsg.setSender(sSend);
		imMsg.setTime(sTime);
		imMsg.setMsg("...");
		
		return imMsg;
	}
	
	
	 
}

