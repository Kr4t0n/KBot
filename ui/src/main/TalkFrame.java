package main;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.EventQueue;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextPane;

import net.miginfocom.swing.MigLayout;
import talk.Initialize;
import talk.MyTalk;
import entities.BubbleModel;
import entities.IMMessage;
import guicomponents.BubbleRenderer;

/**
 * 
 * @author fengyuze
 *
 */
public class TalkFrame extends JFrame {
	
	private static final long serialVersionUID = 1L;
	final JTextPane txtPnl = new JTextPane();
	JButton btnSend = new JButton("Send");
	JScrollPane scrollPane;
	JTable table;
	String userSay;
	
	BubbleModel mModel = new BubbleModel();
	
	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					TalkFrame frame = new TalkFrame();
					
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the frame.
	 */
	public TalkFrame() {
		initGUI();
		this.setVisible(true);
		new Thread() {
			@Override
			public void run() {
				new Initialize(TalkFrame.this);
			}
		}.start();
		
	}
	
	public void initGUI() {
		setTitle("My Chatbot");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 500, 550);
		setLocationRelativeTo(null);
		JPanel contentPane = new JPanel();
		contentPane.setLayout(new BorderLayout(0, 0));
		setContentPane(contentPane);
		
		scrollPane = new JScrollPane();
		contentPane.add(scrollPane, BorderLayout.CENTER);
		
		table = new JTable();
		table.setTableHeader(null);
		table.setModel(mModel);
		table.getColumnModel().getColumn(0).setPreferredWidth(260);
		table.getColumnModel().getColumn(0).setCellRenderer(new BubbleRenderer());
		scrollPane.setViewportView(table);
		table.setBackground(Color.white);
		table.setOpaque(true);
		table.setShowHorizontalLines(false);
		scrollPane.getViewport().setBackground(Color.WHITE);
		
		
		
		
		JPanel pnlSend = new JPanel(new MigLayout("ins 4"));
		pnlSend.add(new JScrollPane(txtPnl), "hmin 50px,growx,pushx");
		pnlSend.add(btnSend, "growy,pushy");
		contentPane.add(pnlSend, BorderLayout.SOUTH);
		btnSend.setEnabled(false);
		btnSend.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				String sMsg = txtPnl.getText().trim();
				if ("".equals(sMsg)) {
					System.err.println("Require not blank.");
					return ;
				}
				String sSend = "User";
				String sTime =  getTime();
				IMMessage imMsg = new IMMessage();
				imMsg.setSender(sSend);
				imMsg.setTime(sTime);
				imMsg.setMsg(sMsg);
				mModel.addRow(imMsg);
				userSay=sMsg;
				//clear
				txtPnl.setText("");
				btnSend.setEnabled(false);
				new resThread().start();
				
				
			}
		});
		String sSend = "Chatbot";
		String sTime =  getTime();
		IMMessage imMsg = new IMMessage();
		imMsg.setSender(sSend);
		imMsg.setTime(sTime);
		imMsg.setMsg("Loading...Please wait!");
		mModel.addRow(imMsg);
		
	}
	
	
	public void enableGUI() {
		String sSend = "Chatbot";
		String sTime =  getTime();
		IMMessage imMsg = new IMMessage();
		imMsg.setSender(sSend);
		imMsg.setTime(sTime);
		imMsg.setMsg("We can start conversation now!");
		mModel.addRow(imMsg);
		//clear
		btnSend.setEnabled(true);
	}
	static String getTime() {
		SimpleDateFormat sdf = new SimpleDateFormat("hh:mm:ss");
		return sdf.format(new Date());
	}
	
	private class resThread extends Thread{
		@Override 
		public void run() {
			IMMessage imMsg = new MyTalk().say(userSay);
			mModel.addRow(imMsg);
			//clear
			btnSend.setEnabled(true);
			
			
			
			

			
		}
	}
	
}
