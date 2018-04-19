package entities;

import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;
/**
 * 
 * @author fengyuze
 *
 */
public class BubbleModel extends AbstractTableModel{
	
	private static final long serialVersionUID = 1L;
	
	private List<IMMessage> mLstData;
	
	public BubbleModel() {
		mLstData = new ArrayList<IMMessage>();
	}
	
	public void addRow(IMMessage imMsg) {
		mLstData.add(imMsg);
		int iLen = mLstData.size();
		this.fireTableRowsInserted(iLen-1, iLen-1);
	}
	
	@Override
	public int getRowCount() {
		return mLstData.size();
	}

	@Override
	public int getColumnCount() {
		return 1;
	}

	@Override
	public Object getValueAt(int rowIndex, int columnIndex) {
		return mLstData.get(rowIndex);
	}
	
	
}
