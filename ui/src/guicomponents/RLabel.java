package guicomponents;
import java.awt.Graphics;
import java.awt.Insets;

import javax.swing.Icon;
import javax.swing.JLabel;


/**
 * 
 * @author fengyuze
 *
 */
public class RLabel extends JLabel{
	
	private static final long serialVersionUID = 1L;
	
	private Icon mBgIcon;
	
	public void setBackgroundIcon(Icon icon) {
		mBgIcon = icon;
	}
	
	@Override
	protected void paintComponent(Graphics g) {
		if (mBgIcon != null) {
			mBgIcon.paintIcon(this, g, 0, 0);
		}
		super.paintComponent(g);
	}
	
	@Override
	public Insets getInsets() {
		return new Insets(10,10,10,10);
	}
	
}
